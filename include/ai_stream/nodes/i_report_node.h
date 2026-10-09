// include/ai_stream/nodes/i_report_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <string>
#include <vector>

namespace ai_stream {
namespace nodes {

/**
 * @brief 告警上报出口
 *
 * 上游是谁就上报谁：没接 vlm_gate 时上游是 evidence（按告警直接上报）；
 * 接了 gate 时上游是 vlm_gate（只收到判真的事件）。本节点自身不含条件逻辑，
 * 靠管道接线切换，语义天然满足"接入 gate 后只上报判真告警"。
 */
struct ReportConfig {
    bool enabled = true;
    std::string url;                       // webhook 地址
    std::string api_key;                   // 支持 ${ENV}，默认明文
    std::vector<std::string> headers;       // 额外请求头，如 "X-Tenant: a"
    long timeout_ms = 3000;
    long connect_timeout_ms = 1500;
    int retries = 1;
    std::string source = "ai_stream";      // payload 里标识来源

    // 报警图 base64 附带
    bool attach_image = true;              // 关闭则只发结构化告警，不带图
    int image_max_side = 0;                // >0 时按最长边下采样再编码，控 payload 体积；0=原样
    std::string image_field = "snapshot_base64";  // 图片在 payload 中的字段名
};

class IReportNode : public core::Node {
public:
    using core::Node::Node;

    virtual void setReportConfig(const ReportConfig& config) = 0;

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        (void)node_id;
        ReportConfig c;
        c.enabled = params.value("enabled", c.enabled);
        c.url = params.value("url", c.url);
        c.api_key = params.value("api_key", c.api_key);
        c.timeout_ms = params.value("timeout_ms", c.timeout_ms);
        c.connect_timeout_ms = params.value("connect_timeout_ms", c.connect_timeout_ms);
        c.retries = params.value("retries", c.retries);
        c.source = params.value("source", c.source);
        c.attach_image = params.value("attach_image", c.attach_image);
        c.image_max_side = params.value("image_max_side", c.image_max_side);
        c.image_field = params.value("image_field", c.image_field);
        if (params.contains("headers") && params["headers"].is_array())
            c.headers = params["headers"].get<std::vector<std::string>>();

        if (c.enabled && c.url.empty()) {
            LOG_ERROR_FMT("[Report] 'url' is required when enabled");
            return false;
        }
        setReportConfig(c);
        return true;
    }
};

}  // namespace nodes
}  // namespace ai_stream
