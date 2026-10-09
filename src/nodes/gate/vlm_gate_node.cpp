// src/nodes/gate/vlm_gate_node.cpp
#include "vlm_gate_node.h"
#include "registry/node_factory.h"
#include "http_util.h"
#include "utils/time_util.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;
using nlohmann::json;

namespace ai_stream {
namespace nodes {

namespace {

// 从可能带 ```json 围栏 / 前后噪声的 content 里抠出第一个 JSON 对象。
std::string extractJsonObject(const std::string& text)
{
    const size_t l = text.find('{');
    const size_t r = text.rfind('}');
    if (l == std::string::npos || r == std::string::npos || r < l)
        return {};
    return text.substr(l, r - l + 1);
}

// 宽容取 bool：true/false 可能是 bool、"true"、1/0。
bool jsonToBool(const json& j, const char* key, bool& out)
{
    if (!j.contains(key))
        return false;
    const json& v = j[key];
    if (v.is_boolean()) { out = v.get<bool>(); return true; }
    if (v.is_number())   { out = v.get<double>() >= 0.5; return true; }
    if (v.is_string()) {
        std::string s = v.get<std::string>();
        std::transform(s.begin(), s.end(), s.begin(),
                       [](unsigned char c) { return std::tolower(c); });
        out = (s == "true" || s == "yes" || s == "1");
        return true;
    }
    return false;
}

double jsonToDouble(const json& j, const char* key, double def)
{
    if (!j.contains(key)) return def;
    const json& v = j[key];
    if (v.is_number()) return v.get<double>();
    if (v.is_string()) { try { return std::stod(v.get<std::string>()); } catch (...) {} }
    return def;
}

std::string dedupKey(const core::InferenceResultPacket& pkt, const rules::AlertEvent& ev)
{
    const int obj = ev.object_ids.empty() ? -1 : ev.object_ids.front();
    return std::to_string(pkt.stream_id) + "|" +
           std::to_string(static_cast<int>(ev.alert_type)) + "|" +
           std::to_string(ev.zone_no) + "|" + std::to_string(obj);
}

}  // namespace

VlmGateNode::VlmGateNode() : core::QueuedNode<IVlmGateNode>("VlmGateNode") {}

VlmGateNode::~VlmGateNode() { core::QueuedNode<IVlmGateNode>::stop(); }

void VlmGateNode::setGateConfig(const VlmGateConfig& config)
{
    cfg_ = config;
    cfg_.api_key = http::expandEnv(cfg_.api_key);   // 明文默认，但支持 ${ENV}
}

bool VlmGateNode::onStartup()
{
    if (!cfg_.enabled) {
        LOG_INFO("[VlmGate] disabled");
        return true;
    }
    std::error_code ec;
    fs::create_directories(cfg_.review_dir, ec);
    if (ec)
        LOG_WARN_FMT("[VlmGate] cannot create review dir {}: {}", cfg_.review_dir, ec.message());
    review_path_ = (fs::path(cfg_.review_dir) / "decisions.jsonl").string();
    LOG_INFO_FMT("[VlmGate] ready url={} model={} tau={} cooldown={}s",
                 cfg_.url, cfg_.model, cfg_.confidence_threshold, cfg_.cooldown_s);
    return true;
}

std::string VlmGateNode::loadSnapshotBase64(const std::string& path, int max_side)
{
    if (path.empty())
        return {};
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
    if (img.empty()) {
        LOG_WARN_FMT("[VlmGate] snapshot unreadable: {}", path);
        return {};
    }
    const int longest = std::max(img.cols, img.rows);
    if (max_side > 0 && longest > max_side) {
        const double s = static_cast<double>(max_side) / longest;
        cv::resize(img, img, cv::Size(), s, s, cv::INTER_AREA);
    }
    std::vector<unsigned char> buf;
    if (!cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, 85}))
        return {};
    return http::base64Encode(buf);
}

namespace {

// alert_name → 人可读：child_near_wall → "child near wall"（喂给模型的 {alert}）
std::string prettifyAlertName(const std::string& name)
{
    std::string s = name;
    std::replace(s.begin(), s.end(), '_', ' ');
    std::replace(s.begin(), s.end(), '-', ' ');
    return s;
}

std::string fillTemplate(const std::string& tpl, const std::string& alert)
{
    std::string out;
    out.reserve(tpl.size());
    for (size_t i = 0; i < tpl.size();) {
        if (tpl[i] == '{' && tpl.compare(i, 7, "{alert}") == 0) {
            out += alert;
            i += 7;
        } else {
            out += tpl[i++];
        }
    }
    return out;
}

}  // namespace

VlmGateNode::GateMode VlmGateNode::resolvePrompt(const std::string& alert_name,
                                                 VlmPromptSpec& out)
{
    auto it = cfg_.prompts.find(alert_name);
    if (it != cfg_.prompts.end()) {
        out = it->second;
        if (out.system.empty()) out.system = cfg_.system_prompt;
        if (!out.has_threshold) { out.confidence_threshold = cfg_.confidence_threshold; }
        if (!out.has_max_tokens) { out.max_tokens = cfg_.max_tokens; }
        return GateMode::Verify;
    }

    // 未命中：按策略
    switch (cfg_.unmatched_policy) {
        case VlmUnmatchedPolicy::Passthrough: return GateMode::Passthrough;
        case VlmUnmatchedPolicy::Hold:        return GateMode::Hold;
        case VlmUnmatchedPolicy::Auto:
        default:
            // 用 alert_name 套模板自动生成核验问题
            out = VlmPromptSpec{};
            out.question = fillTemplate(cfg_.question_template, prettifyAlertName(alert_name));
            out.system = cfg_.system_prompt;
            out.confidence_threshold = cfg_.confidence_threshold;
            out.max_tokens = cfg_.max_tokens;
            return GateMode::Verify;
    }
}

VlmGateNode::Verdict VlmGateNode::callVlm(const std::string& image_b64, const VlmPromptSpec& spec)
{
    Verdict v;
    const long long t0 = utils::TimeUtil::currentTimeMs();
    const std::string url = cfg_.url + cfg_.endpoint;

    // OpenAI 兼容 body：system 只输出 JSON，user 带"核验问题 + 标注图"。
    json question = json{{"type", "text"}, {"text", spec.question}};
    json image = json{{"type", "image_url"},
                      {"image_url", {{"url", "data:image/jpeg;base64," + image_b64}}}};
    json body = json{
        {"model", cfg_.model},
        {"temperature", 0},
        {"max_tokens", spec.max_tokens > 0 ? spec.max_tokens : cfg_.max_tokens},
        {"messages",
         {{{"role", "system"}, {"content", spec.system}},
          {{"role", "user"}, {"content", {question, image}}}}},
    };
    // 关思考：vLLM/Qwen 经 chat_template_kwargs 透传。思考模型会先吐很长的
    // reasoning_content，小 max_tokens 下可见 content 直接为空 → 解析失败 → HOLD。
    if (!cfg_.enable_thinking)
        body["chat_template_kwargs"] = json{{"enable_thinking", false}};

    std::vector<std::string> headers;
    if (!cfg_.api_key.empty())
        headers.push_back("Authorization: Bearer " + cfg_.api_key);

    http::Response resp;
    const int attempts = 1 + std::max(0, cfg_.retries);
    for (int i = 0; i < attempts; ++i) {
        resp = http::postJson(url, body.dump(), headers, cfg_.timeout_ms, cfg_.connect_timeout_ms);
        if (resp.ok())
            break;
    }

    v.latency_ms = utils::TimeUtil::currentTimeMs() - t0;

    if (!resp.ok()) {
        v.error = resp.err.empty() ? ("http " + std::to_string(resp.status)) : resp.err;
        return v;   // error → 上层按 HOLD 处理，不丢弃也不上报
    }

    // 解析 choices[0].message.content
    try {
        json root = json::parse(resp.body);
        const json& msg = root["choices"][0]["message"];
        const json& content = msg["content"];
        // 思考模型常见：content 为 null（token 被 reasoning_content 吃光）。
        if (content.is_null() || (content.is_string() && content.get<std::string>().empty())) {
            std::string rc = msg.contains("reasoning_content") && msg["reasoning_content"].is_string()
                                 ? msg["reasoning_content"].get<std::string>() : "";
            v.error = "empty content (thinking model? set enable_thinking=false or raise max_tokens)"
                      "; reasoning_len=" + std::to_string(rc.size());
            LOG_WARN_FMT("[VlmGate] 模型返回空 content，reasoning_len={}", rc.size());
            return v;
        }
        std::string text =
            content.is_string() ? content.get<std::string>() : content.dump();
        const std::string obj = extractJsonObject(text);
        if (obj.empty()) {
            v.error = "no json in content";
            LOG_WARN_FMT("[VlmGate] content 无 JSON，原文(截断): {}", text.substr(0, 200));
            return v;
        }
        json pj = json::parse(obj);
        bool verdict = false;
        if (!jsonToBool(pj, "verdict", verdict)) {
            v.error = "missing verdict field";
            LOG_WARN_FMT("[VlmGate] content 缺 verdict，原文(截断): {}", obj.substr(0, 200));
            return v;
        }
        v.verdict = verdict;
        v.confidence = static_cast<float>(jsonToDouble(pj, "confidence", verdict ? 1.0 : 0.0));
        if (pj.contains("reason") && pj["reason"].is_string())
            v.reason = pj["reason"].get<std::string>();
    } catch (const std::exception& e) {
        v.error = std::string("parse fail: ") + e.what();
        LOG_WARN_FMT("[VlmGate] 响应解析失败: {} body(截断): {}", e.what(),
                     resp.body.substr(0, 240));
    }
    return v;
}

bool VlmGateNode::inCooldown(const std::string& key, long long now_ms)
{
    std::lock_guard<std::mutex> lk(cooldown_mutex_);
    auto it = last_decision_ms_.find(key);
    if (it == last_decision_ms_.end())
        return false;
    return now_ms - it->second < cfg_.cooldown_s * 1000LL;
}

void VlmGateNode::touchCooldown(const std::string& key, long long now_ms)
{
    std::lock_guard<std::mutex> lk(cooldown_mutex_);
    last_decision_ms_[key] = now_ms;
}

void VlmGateNode::appendAudit(const core::InferenceResultPacket& pkt, const rules::AlertEvent& ev,
                              const std::string& snap_path, const Verdict& v)
{
    if (review_path_.empty())
        return;
    json rec;
    rec["ts_ms"] = utils::TimeUtil::currentTimeMs();
    rec["stream_id"] = pkt.stream_id;
    rec["alert_id"] = ev.alert_id;
    rec["alert_name"] = ev.alert_name;
    rec["zone_no"] = ev.zone_no;
    rec["object_ids"] = ev.object_ids;
    rec["snapshot_path"] = snap_path;
    rec["decision"] = v.decision;
    rec["vlm_verdict"] = v.verdict;
    rec["vlm_confidence"] = v.confidence;
    rec["vlm_reason"] = v.reason;
    rec["vlm_error"] = v.error;
    rec["latency_ms"] = v.latency_ms;
    rec["model"] = cfg_.model;
    rec["ground_truth"] = nullptr;   // 人工复核回填，用于回流评测/微调

    std::ofstream out(review_path_, std::ios::app);
    if (out)
        out << rec.dump() << "\n";
}

void VlmGateNode::processPacket(std::shared_ptr<core::BasePacket> packet)
{
    if (!packet)
        return;
    if (packet->type == core::PacketType::STREAM_END) {
        broadcast(packet);
        return;
    }
    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }

    auto infer = std::static_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer)
        return;

    // 未启用 / 关闭核验：直接透传（等价于没接 gate）。
    if (!cfg_.enabled || cfg_.passthrough) {
        broadcast(infer);
        return;
    }

    const long long now = utils::TimeUtil::currentTimeMs();
    bool any_push = false;

    for (auto& result : infer->alert_result) {
        std::vector<rules::AlertEvent> keep;
        for (auto& ev : result.alert_events) {
            const std::string key = dedupKey(*infer, ev);
            const std::string snap = ev.extra_data.value("snapshot_path", "");

            VlmPromptSpec spec;
            const GateMode mode = resolvePrompt(ev.alert_name, spec);

            Verdict v;
            v.pushed = false;

            if (mode == GateMode::Passthrough) {
                // 未配模板且策略=放行：不调 VLM，直接上报
                v.decision = "PASSTHROUGH";
                v.pushed = true;
            } else if (mode == GateMode::Hold) {
                // 未配模板且策略=转人工：不上报、不丢弃
                v.decision = "HOLD";
                v.error = "no_prompt_configured";
            } else {
                // Verify
                if (inCooldown(key, now)) {
                    continue;   // 冷却期内不重复调 VLM，也不重复放行
                }
                if (snap.empty()) {
                    v.error = "no_snapshot";
                    v.decision = "HOLD";
                } else {
                    const std::string b64 = loadSnapshotBase64(snap, cfg_.max_side);
                    if (b64.empty()) {
                        v.error = "image_unreadable";
                        v.decision = "HOLD";
                    } else {
                        v = callVlm(b64, spec);
                        if (!v.error.empty()) {
                            v.decision = "HOLD";               // 异常不判假：转人工
                        } else if (!v.verdict) {
                            v.decision = "DROP";               // 明确判假：不上报
                        } else if (v.confidence >= spec.confidence_threshold) {
                            v.decision = "PUSH";               // 判真且达该类阈值
                            v.pushed = true;
                        } else {
                            v.decision = "HOLD";               // 判真但低置信：转人工
                        }
                    }
                }
                touchCooldown(key, now);
            }

            // 把核验结论挂回事件，供 report / 下游与审计使用
            ev.extra_data["vlm"] = json{
                {"decision", v.decision}, {"verdict", v.verdict},
                {"confidence", v.confidence}, {"reason", v.reason},
                {"error", v.error}, {"model", cfg_.model},
                {"latency_ms", v.latency_ms}};

            appendAudit(*infer, ev, snap, v);

            if (v.pushed) {
                keep.push_back(ev);
                any_push = true;
            }
            LOG_INFO_FMT("[VlmGate] {} decision={} verdict={} conf={:.2f} err={}",
                         ev.alert_name, v.decision, v.verdict, v.confidence, v.error);
        }
        result.alert_events = std::move(keep);
    }

    // 只把"有可上报事件(PUSH/PASSTHROUGH)"的告警下发；全 DROP/HOLD 时不发。
    if (any_push)
        broadcast(infer);
}

REGISTER_NODE("vlm_gate", VlmGateNode)

}  // namespace nodes
}  // namespace ai_stream
