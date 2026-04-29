// src/nodes/draw/osd_draw.cpp
#include "osd_draw.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace ai_stream {
namespace nodes {

OSDDrawNode::OSDDrawNode() : IDrawNode("OSDDraw") {}

void OSDDrawNode::setBoxColor(int b, int g, int r) {
    box_color_ = cv::Scalar(b, g, r);
}

void OSDDrawNode::setFontThickness(int thickness) {
    font_thickness_ = thickness;
}

void OSDDrawNode::setShowConfidence(bool show) {
    show_confidence_ = show;
}

void OSDDrawNode::setClassFilter(const std::vector<int>& class_ids) {
    class_filter_ = class_ids;
}

void OSDDrawNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }

    auto infer_result = std::static_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer_result->source_frame || !infer_result->source_frame->mat) {
        LOG_WARN_FMT("[OSDDraw] Inference result missing source frame");
        return;
    }

    // 克隆一份图像避免影响其他分支（如果该帧还要被其他节点使用）
    auto draw_mat = std::make_shared<cv::Mat>(infer_result->source_frame->source_mat->clone());
    
    for (const auto& det : infer_result->detections) {
        // 类别过滤
        if (!class_filter_.empty() && 
            std::find(class_filter_.begin(), class_filter_.end(), det.class_id) == class_filter_.end()) {
            continue;
        }

        cv::Rect rect(static_cast<int>(det.x), static_cast<int>(det.y),
                      static_cast<int>(det.w), static_cast<int>(det.h));
        LOG_DEBUG_FMT("[OSDDraw] Drawing detection: {} {} ({}, {}, {}, {})",std::to_string(det.class_id).c_str(), det.confidence, rect.x, rect.y, rect.width, rect.height);
        cv::rectangle(*draw_mat, rect, box_color_, font_thickness_);

        std::string label = det.class_name;
        std::string track_info = "ID:" + std::to_string(det.track_id);
        if (show_confidence_) {
            label += " " + std::to_string(det.confidence).substr(0, 4);
        }
        if (det.track_id >= 0) {
            label += " " + track_info;
        }
        cv::putText(*draw_mat, label, 
                    cv::Point(rect.x, rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color_, font_thickness_);
        LOG_DEBUG_FMT("[OSDDraw] Label: {}, track_id: {}", label.c_str(), det.track_id);
    }

    LOG_DEBUG_FMT("[OSDDraw] Drawing {} detections", infer_result->detections.size());

    // 构造新的视频帧包（包含绘制后的图像）
    auto drawn_frame = std::make_shared<core::VideoFramePacket>();
    drawn_frame->stream_id = infer_result->stream_id;
    drawn_frame->timestamp_ms = infer_result->timestamp_ms;
    drawn_frame->mat = draw_mat;
    drawn_frame->width = draw_mat->cols;
    drawn_frame->height = draw_mat->rows;
    drawn_frame->channels = draw_mat->channels();

    broadcast(drawn_frame);
}

REGISTER_NODE("osd_draw", OSDDrawNode)

} // namespace nodes
} // namespace ai_stream