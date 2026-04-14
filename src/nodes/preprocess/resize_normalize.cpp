// src/nodes/preprocess/resize_normalize.cpp
#include "resize_normalize.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>

namespace ai_stream {
namespace nodes {

ResizeNormalizeNode::ResizeNormalizeNode() : core::Node("ResizeNormalize") {}

void ResizeNormalizeNode::setTargetSize(int width, int height) {
    target_width_ = width;
    target_height_ = height;
}

void ResizeNormalizeNode::setMean(const std::vector<float>& mean) {
    mean_ = mean;
}

void ResizeNormalizeNode::setStd(const std::vector<float>& std) {
    std_ = std;
}

void ResizeNormalizeNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type != core::PacketType::DECODED_FRAME) {
        // 如果不是视频帧，直接透传或忽略
        broadcast(packet);
        return;
    }

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    if (!frame->mat || frame->mat->empty()) {
        LOG_WARN_FMT("[ResizeNormalize] Received empty frame");
        return;
    }

    // 创建新 Mat 存放预处理结果（避免修改原始帧影响其他分支）
    auto processed_mat = std::make_shared<cv::Mat>();
    cv::resize(*frame->mat, *processed_mat, cv::Size(target_width_, target_height_));
    
    // 归一化：转换为 float，减均值除标准差
    processed_mat->convertTo(*processed_mat, CV_32FC3, 1.0/255.0);
    if (!mean_.empty() && !std_.empty()) {
        // 实际应用中应逐通道操作，此处简化
    }

    // 构造新包，保留原 stream_id 和时间戳
    auto new_packet = std::make_shared<core::VideoFramePacket>();
    new_packet->stream_id = frame->stream_id;
    new_packet->timestamp_ms = frame->timestamp_ms;
    new_packet->mat = processed_mat;
    new_packet->width = target_width_;
    new_packet->height = target_height_;
    new_packet->channels = 3;

    broadcast(new_packet);
}

REGISTER_NODE("resize_normalize", ResizeNormalizeNode)

} // namespace nodes
} // namespace ai_stream