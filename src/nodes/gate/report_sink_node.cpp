// src/nodes/gate/report_sink_node.cpp
#include "report_sink_node.h"
#include "registry/node_factory.h"
#include "http_util.h"
#include "utils/time_util.h"

#include <nlohmann/json.hpp>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <fstream>
#include <iterator>
#include <vector>

using nlohmann::json;

namespace ai_stream {
namespace nodes {

namespace {
// 读快照并转 base64。max_side>0 先按最长边下采样重编码，控 payload 体积；
// 否则原样读文件字节编码（不重编码，避免画质损失）。失败返回 false。
bool loadImageBase64(const std::string& path, int max_side, std::string& out_b64)
{
    out_b64.clear();
    if (path.empty())
        return false;

    if (max_side > 0) {
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        if (img.empty())
            return false;
        const int longest = std::max(img.cols, img.rows);
        if (longest > max_side) {
            const double s = static_cast<double>(max_side) / longest;
            cv::resize(img, img, cv::Size(), s, s, cv::INTER_AREA);
        }
        std::vector<unsigned char> buf;
        if (!cv::imencode(".jpg", img, buf, {cv::IMWRITE_JPEG_QUALITY, 85}))
            return false;
        out_b64 = http::base64Encode(buf);
        return !out_b64.empty();
    }

    std::ifstream f(path, std::ios::binary);
    if (!f)
        return false;
    std::vector<unsigned char> buf((std::istreambuf_iterator<char>(f)),
                                   std::istreambuf_iterator<char>());
    if (buf.empty())
        return false;
    out_b64 = http::base64Encode(buf);
    return !out_b64.empty();
}
}  // namespace

ReportSinkNode::ReportSinkNode() : core::QueuedNode<IReportNode>("ReportSinkNode") {}

ReportSinkNode::~ReportSinkNode() { core::QueuedNode<IReportNode>::stop(); }

void ReportSinkNode::setReportConfig(const ReportConfig& config)
{
    cfg_ = config;
    cfg_.api_key = http::expandEnv(cfg_.api_key);   // 明文默认，允许 ${ENV}
}

bool ReportSinkNode::onStartup()
{
    if (cfg_.enabled && cfg_.url.empty()) {
        LOG_ERROR("[Report] enabled but url empty");
        return false;
    }
    LOG_INFO_FMT("[Report] ready url={} source={}", cfg_.url, cfg_.source);
    return true;
}

void ReportSinkNode::processPacket(std::shared_ptr<core::BasePacket> packet)
{
    if (!packet)
        return;
    if (packet->type == core::PacketType::STREAM_END) {
        broadcast(packet);   // 终端节点也需把 STREAM_END 透传（本无下游则空转）
        return;
    }
    if (packet->type != core::PacketType::META_DATA || !cfg_.enabled)
        return;

    auto infer = std::static_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer)
        return;

    std::vector<std::string> headers = cfg_.headers;
    if (!cfg_.api_key.empty())
        headers.push_back("Authorization: Bearer " + cfg_.api_key);

    for (const auto& result : infer->alert_result) {
        for (const auto& ev : result.alert_events) {
            // 到这里的事件已是"可上报"的（无 gate 时为全部告警，有 gate 时为判真告警）
            json payload;
            payload["source"] = cfg_.source;
            payload["report_ts_ms"] = utils::TimeUtil::currentTimeMs();
            payload["stream_id"] = infer->stream_id;
            payload["timestamp_ms"] = infer->timestamp_ms;
            payload["alert"] = ev.toJson();   // 含 snapshot_path 与 vlm 结论(extra_data)

            // 附带报警图（base64）：取 evidence 落的**已标注**快照，而不是原始帧。
            // 缺图/读失败只降级为"无图上报"，不因图片问题丢掉告警本身。
            std::string img_b64;
            size_t img_bytes = 0;
            if (cfg_.attach_image) {
                const std::string snap = ev.extra_data.value("snapshot_path", "");
                if (loadImageBase64(snap, cfg_.image_max_side, img_b64)) {
                    payload[cfg_.image_field] = img_b64;
                    payload["image_mime"] = "image/jpeg";
                    img_bytes = img_b64.size();
                } else if (!snap.empty()) {
                    LOG_WARN_FMT("[Report] snapshot 无法编码为 base64，改为无图上报: {}", snap);
                } else {
                    LOG_WARN("[Report] 告警无 snapshot_path，改为无图上报");
                }
            }

            http::Response resp;
            const int attempts = 1 + std::max(0, cfg_.retries);
            for (int i = 0; i < attempts; ++i) {
                resp = http::postJson(cfg_.url, payload.dump(), headers,
                                      cfg_.timeout_ms, cfg_.connect_timeout_ms);
                if (resp.ok())
                    break;
            }

            if (resp.ok()) {
                LOG_INFO_FMT("[Report] pushed alert '{}' (http {}) image={} b64={}B body={}B",
                             ev.alert_name, resp.status, img_bytes ? "yes" : "no",
                             img_bytes, payload.dump().size());
            } else {
                LOG_WARN_FMT("[Report] push FAILED alert '{}' err={} status={} body={}",
                             ev.alert_name, resp.err, resp.status,
                             resp.body.substr(0, 200));
            }
        }
    }
}

REGISTER_NODE("report", ReportSinkNode)

}  // namespace nodes
}  // namespace ai_stream
