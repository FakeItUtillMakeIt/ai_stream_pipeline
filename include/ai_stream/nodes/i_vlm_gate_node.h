// include/ai_stream/nodes/i_vlm_gate_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <string>

namespace ai_stream {
namespace nodes {

/**
 * @brief VLM 告警闸门配置
 *
 * 本地与云端统一走一个 OpenAI 兼容端点（url + key + model），不区分后端。
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

    float confidence_threshold = 0.6f;   // verdict=true 且 conf>=τ 才放行
    long cooldown_s = 30;                // 同一目标冷却窗口，防重复调用

    std::string review_dir = "./vlm_review";   // HOLD/DROP/error 审计与人工回流
    // true 时直接放行（等价于不接 gate 的行为），用于灰度回退
    bool passthrough = false;
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
        c.confidence_threshold = params.value("confidence_threshold", c.confidence_threshold);
        c.cooldown_s = params.value("cooldown_s", c.cooldown_s);
        c.review_dir = params.value("review_dir", c.review_dir);
        c.passthrough = params.value("passthrough", c.passthrough);

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