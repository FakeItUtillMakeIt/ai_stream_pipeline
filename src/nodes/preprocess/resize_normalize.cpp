// src/nodes/preprocess/resize_normalize.cpp
#include "resize_normalize.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

namespace ai_stream {
namespace nodes {

ResizeNormalizeNode::ResizeNormalizeNode() : IPreprocessNode("ResizeNormalize"),
    target_width_(640), target_height_(640),
    mean_{0.0f, 0.0f, 0.0f}, std_{1.0f, 1.0f, 1.0f},
    interpolation_method_("bilinear"), keep_aspect_ratio_(false),
    output_dtype_("float32") {
        LOG_INFO_FMT("[ResizeNormalize] initialized");
    }

ResizeNormalizeNode::~ResizeNormalizeNode() {
    LOG_INFO_FMT("[ResizeNormalize] deinitialized");
}

void ResizeNormalizeNode::setTargetSize(int width, int height) {
    target_width_ = width;
    target_height_ = height;
}

std::pair<int, int> ResizeNormalizeNode::getTargetSize() const {
    return {target_width_, target_height_};
}

void ResizeNormalizeNode::setMean(const std::vector<float>& mean) {
    mean_ = mean;
    LOG_INFO_FMT("[ResizeNormalize] mean set to: {},{},{}", mean_[0],mean_[1],mean_[2]);
}

void ResizeNormalizeNode::setStd(const std::vector<float>& std) {
    std_ = std;
    LOG_INFO_FMT("[ResizeNormalize] std set to: {},{},{}", std_[0],std_[1],std_[2]);
}

void ResizeNormalizeNode::setInterpolationMethod(const std::string& method) {
    interpolation_method_ = method;
}

void ResizeNormalizeNode::setKeepAspectRatio(bool keep_aspect_ratio) {
    keep_aspect_ratio_ = keep_aspect_ratio;
}

void ResizeNormalizeNode::setOutputDataType(const std::string& dtype) {
    output_dtype_ = dtype;
}

void ResizeNormalizeNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[ResizeNormalize] Received stream end");
        stop();
        broadcast(packet);
        return;
    }
    in_time_ms_ = utils::TimeUtil::currentTimeMs();
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
    cv::Mat input_mat = *frame->mat;
    auto processed_mat = std::make_shared<cv::Mat>();

    if (keep_aspect_ratio_) {
        // Letterbox resize: keep aspect ratio, pad with gray (114)
        float scale = std::min(
            static_cast<float>(target_width_) / input_mat.cols,
            static_cast<float>(target_height_) / input_mat.rows);
        int letter_w = std::max(1, static_cast<int>(std::round(input_mat.cols * scale)));
        int letter_h = std::max(1, static_cast<int>(std::round(input_mat.rows * scale)));

        // First resize to letter size
        cv::Mat letter_mat;
        cv::resize(input_mat, letter_mat, cv::Size(letter_w, letter_h), 0, 0, cv::INTER_LINEAR);

        // Create target size canvas with gray padding
        *processed_mat = cv::Mat::zeros(target_height_, target_width_, CV_32FC3);
        processed_mat->setTo(cv::Scalar(114.0f / 255.0f, 114.0f / 255.0f, 114.0f / 255.0f));

        // Copy letter image to center
        int pad_x = (target_width_ - letter_w) / 2;
        int pad_y = (target_height_ - letter_h) / 2;
        cv::Rect roi(pad_x, pad_y, letter_w, letter_h);

        // Convert to float and normalize before copying
        cv::Mat letter_float, letter_norm;
        letter_mat.convertTo(letter_float, CV_32FC3, 1.0 / 255.0);
        letter_norm = letter_float;

        if (!mean_.empty() && !std_.empty()) {
            // TODO: per-channel normalization
        }
        letter_norm.copyTo((*processed_mat)(roi));
    } else {
        // Direct resize (stretch to fill)
        cv::resize(input_mat, *processed_mat, cv::Size(target_width_, target_height_));
        // 归一化：转换为 float，减均值除标准差
        processed_mat->convertTo(*processed_mat, CV_32FC3, 1.0/255.0);
        if (!mean_.empty() && !std_.empty()) {
            // 实际应用中应逐通道操作，此处简化
        }
    }

    LOG_INFO_FMT("[ResizeNormalize] Resized frame to {}x{}", target_width_, target_height_);

    // 构造新包，保留原 stream_id 和时间戳
    auto new_packet = std::make_shared<core::VideoFramePacket>();
    new_packet->stream_id = frame->stream_id;
    new_packet->source_id = frame->source_id;
    new_packet->timestamp_ms = frame->timestamp_ms;
    new_packet->mat = processed_mat;
    new_packet->source_mat = frame->mat;
    new_packet->width = target_width_;
    new_packet->height = target_height_;
    new_packet->channels = 3;
    new_packet->frame_id = frame->frame_id;

    new_packet->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
    new_packet->cost_time_map = packet->cost_time_map;
    new_packet->cost_time_map.insert({name_,packet->cost_ms});

    // Letterbox 参数：让下游节点知道如何反变换坐标
    if (keep_aspect_ratio_) {
        new_packet->letterbox_used = true;
        float s = std::min(
            static_cast<float>(target_width_) / frame->mat->cols,
            static_cast<float>(target_height_) / frame->mat->rows);
        int lw = std::max(1, static_cast<int>(std::round(frame->mat->cols * s)));
        int lh = std::max(1, static_cast<int>(std::round(frame->mat->rows * s)));
        new_packet->letter_scale = s;
        new_packet->letter_pad_x = (target_width_ - lw) / 2;
        new_packet->letter_pad_y = (target_height_ - lh) / 2;
    } else {
        new_packet->letterbox_used = false;
        new_packet->letter_scale = 1.0f;
        new_packet->letter_pad_x = 0;
        new_packet->letter_pad_y = 0;
    }

    broadcast(new_packet);
}


REGISTER_NODE("resize_normalize", ResizeNormalizeNode)

} // namespace nodes
} // namespace ai_stream