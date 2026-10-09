// include/ai_stream/nodes/i_vlm_gate_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <map>
#include <string>

namespace ai_stream {
namespace nodes {

/**
 * @brief 单个告警类型的核验提示词（可覆盖全局项）
 */
struct VlmPromptSpec {
    std::string question;                       // 完整核验问题
    std::string system;                          // 可选：覆盖全局 system_prompt
    bool has_threshold = false;
    float confidence_threshold = 0.6f;           // 该类型的放行阈值（不填用全局）
    bool has_max_tokens = false;
    int max_tokens = 0;                          // 该类型生成上限（不填用全局）
};

// 未显式配 prompts 的告警怎么处理
enum class VlmUnmatchedPolicy {
    Auto,        // 用 alert_name 套 question_template 自动生成问题（默认）
    Passthrough, // 不调 VLM，直接放行
    Hold         // 不调 VLM，转人工（不上报）
};

/**
 * @brief VLM 告警闸门配置
 *
 * 本地与云端统一走一个 OpenAI 兼容端点（url + key + model），不区分后端。
 * 一套 gate 处理多种告警：按 event.alert_name 查 prompts，命中用其精调问题；
 * 未命中按 unmatched_policy 处理，默认用 alert_name 自动生成核验问题。
 */
struct VlmGateConfig {
    bool enabled = true;
    std::string url;               // OpenAI 兼容基址，如 http://ip:port/v1
    std::string api_key;           // 支持 "${ENV}" 展开，默认明文
    std::string model;             // 模型名
    std::string endpoint = "/chat/completions";

    long timeout_ms = 3000;        // 整体超时
    long connect_timeout_ms = 1500;
    int retries = 1;               // 失败重试次数（不含首次）
    int max_side = 1024;           // 送图前按最长边下采样，控 token
    int max_tokens = 512;          // 生成上限；思考模型 120 会被 reasoning 吃光→content 空
    // 关闭思考模式（vLLM/Qwen 的 chat_template_kwargs.enable_thinking）。
    // 思考模型默认会先输出很长的 reasoning_content，既拖慢又常把"proximity 核验"
    // 过度推理成 false。关掉后更快、且 content 直接是 JSON。
    bool enable_thinking = false;

    float confidence_threshold = 0.6f;   // verdict=true 且 conf>=τ 才放行（全局默认）
    long cooldown_s = 30;                // 同一目标冷却窗口，防重复调用

    std::string review_dir = "./vlm_review";   // HOLD/DROP/error 审计与人工回流
    // true 时直接放行（等价于不接 gate 的行为），用于灰度回退
    bool passthrough = false;

    // 提示词
    std::string system_prompt =
        "你是安防告警核验助手，只输出一个 JSON 对象，不要多余文字。";
    // 未配 prompts 时用它生成问题，{alert} 会被替换成 prettify 后的 alert_name
    std::string question_template =
        "图中红框为相关目标。请判断告警「{alert}」是否属实（该告警由检测/规则产生，"
        "可能有误报）。只判真假，仅输出 {\"verdict\":true/false,\"confidence\":0.0-1.0,"
        "\"reason\":\"不超过20字\"}";
    std::map<std::string, VlmPromptSpec> prompts;   // key = alert_name
    VlmUnmatchedPolicy unmatched_policy = VlmUnmatchedPolicy::Auto;
};

class IVlmGateNode : public core::Node {
public:
    using core::Node::Node;

    virtual void setGateConfig(const VlmGateConfig& config) = 0;

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        (void)node_id;
        VlmGateConfig c;
        c.enabled = params.value("enabled", c.enabled);
        c.url = params.value("url", c.url);
        c.api_key = params.value("api_key", c.api_key);
        c.model = params.value("model", c.model);
        c.endpoint = params.value("endpoint", c.endpoint);
        c.timeout_ms = params.value("timeout_ms", c.timeout_ms);
        c.connect_timeout_ms = params.value("connect_timeout_ms", c.connect_timeout_ms);
        c.retries = params.value("retries", c.retries);
        c.max_side = params.value("max_side", c.max_side);
        c.max_tokens = params.value("max_tokens", c.max_tokens);
        c.enable_thinking = params.value("enable_thinking", c.enable_thinking);
        c.confidence_threshold = params.value("confidence_threshold", c.confidence_threshold);
        c.cooldown_s = params.value("cooldown_s", c.cooldown_s);
        c.review_dir = params.value("review_dir", c.review_dir);
        c.passthrough = params.value("passthrough", c.passthrough);
        c.system_prompt = params.value("system_prompt", c.system_prompt);
        c.question_template = params.value("question_template", c.question_template);

        const std::string um = params.value("unmatched_policy", std::string("auto"));
        if (um == "passthrough") c.unmatched_policy = VlmUnmatchedPolicy::Passthrough;
        else if (um == "hold")   c.unmatched_policy = VlmUnmatchedPolicy::Hold;
        else                     c.unmatched_policy = VlmUnmatchedPolicy::Auto;

        if (params.contains("prompts") && params["prompts"].is_object()) {
            for (auto& [name, pj] : params["prompts"].items()) {
                VlmPromptSpec sp;
                sp.question = pj.value("question", "");
                sp.system = pj.value("system", "");
                if (pj.contains("confidence_threshold")) {
                    sp.has_threshold = true;
                    sp.confidence_threshold = pj["confidence_threshold"].get<float>();
                }
                if (pj.contains("max_tokens")) {
                    sp.has_max_tokens = true;
                    sp.max_tokens = pj["max_tokens"].get<int>();
                }
                if (!sp.question.empty())
                    c.prompts[name] = std::move(sp);
            }
        }

        if (c.enabled && !c.passthrough) {
            if (c.url.empty() || c.model.empty()) {
                LOG_ERROR_FMT("[VlmGate] 'url' and 'model' are required (enabled, not passthrough)");
                return false;
            }
        }
        setGateConfig(c);
        return true;
    }
};

}  // namespace nodes
}  // namespace ai_stream