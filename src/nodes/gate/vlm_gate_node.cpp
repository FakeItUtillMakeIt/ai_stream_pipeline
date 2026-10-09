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

VlmGateNode::Verdict VlmGateNode::callVlm(const std::string& image_b64, const std::string& alert_name)
{
    (void)alert_name;
    Verdict v;
    const long long t0 = utils::TimeUtil::currentTimeMs();
    const std::string url = cfg_.url + cfg_.endpoint;

    // OpenAI 兼容 body：system 定死只输出 JSON，user 带图文本 + 标注图。
    json question =
        json{{"type", "text"},
             {"text", "图中红框为儿童、蓝框为墙或大门。请判断：是否确实有一名儿童紧贴在墙或大门"
                      "旁边（存在坠落/翻越等风险位置）。只判真假，不纠正类别。仅输出 "
                      "{\"verdict\":true/false,\"confidence\":0.0-1.0,\"reason\":\"不超过20字\"}"}};
    json image = json{{"type", "image_url"},
                      {"image_url", {{"url", "data:image/jpeg;base64," + image_b64}}}};
    json body = json{
        {"model", cfg_.model},
        {"temperature", 0},
        {"max_tokens", 120},
        {"messages",
         {{{"role", "system"},
           {"content", "你是安防告警核验助手，只输出一个 JSON 对象，不要多余文字。"}},
          {{"role", "user"}, {"content", {question, image}}}}},
    };

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
        const json& content = root["choices"][0]["message"]["content"];
        std::string text =
            content.is_string() ? content.get<std::string>() : content.dump();
        const std::string obj = extractJsonObject(text);
        if (obj.empty()) {
            v.error = "no json in content";
            return v;
        }
        json pj = json::parse(obj);
        bool verdict = false;
        if (!jsonToBool(pj, "verdict", verdict)) {
            v.error = "missing verdict field";
            return v;
        }
        v.verdict = verdict;
        v.confidence = static_cast<float>(jsonToDouble(pj, "confidence", verdict ? 1.0 : 0.0));
        if (pj.contains("reason") && pj["reason"].is_string())
            v.reason = pj["reason"].get<std::string>();
    } catch (const std::exception& e) {
        v.error = std::string("parse fail: ") + e.what();
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
            if (inCooldown(key, now)) {
                continue;   // 冷却期内不重复调 VLM，也不重复放行
            }

            const std::string snap = ev.extra_data.value("snapshot_path", "");
            Verdict v;
            v.pushed = false;

            if (snap.empty()) {
                v.error = "no_snapshot";
                v.decision = "HOLD";
            } else {
                const std::string b64 = loadSnapshotBase64(snap, cfg_.max_side);
                if (b64.empty()) {
                    v.error = "image_unreadable";
                    v.decision = "HOLD";
                } else {
                    v = callVlm(b64, ev.alert_name);
                    if (!v.error.empty()) {
                        v.decision = "HOLD";           // 异常不判假：转人工，不丢弃
                    } else if (!v.verdict) {
                        v.decision = "DROP";          // 明确判假：不上报
                    } else if (v.confidence >= cfg_.confidence_threshold) {
                        v.decision = "PUSH";
                        v.pushed = true;
                    } else {
                        v.decision = "HOLD";          // 判真但低置信：转人工
                    }
                }
            }

            // 把核验结论挂回事件，供 report / 下游与审计使用
            ev.extra_data["vlm"] = json{
                {"decision", v.decision}, {"verdict", v.verdict},
                {"confidence", v.confidence}, {"reason", v.reason},
                {"error", v.error}, {"model", cfg_.model},
                {"latency_ms", v.latency_ms}};

            touchCooldown(key, now);
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

    // 只把"有被判真事件"的告警下发；全 DROP/HOLD 时不发任何东西（report 收不到即不上报）。
    if (any_push)
        broadcast(infer);
}

REGISTER_NODE("vlm_gate", VlmGateNode)

}  // namespace nodes
}  // namespace ai_stream
