// src/nodes/sink/rtmp_sink.cpp
#include "rtmp_sink.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "encoder_base.h"
#include <opencv2/opencv.hpp>

namespace ai_stream {
namespace nodes {

RTMPSinkNode::RTMPSinkNode() : QueuedNode("RTMPSink") {}

RTMPSinkNode::~RTMPSinkNode() { stop(); }

void RTMPSinkNode::setTarget(const std::string& target) {
    output_url_ = target;
    LOG_INFO_FMT("[RTMPSink] Target: {}", output_url_);
}

void RTMPSinkNode::setEncodingParams(int bitrate, const std::string& encoder) {
    if (bitrate > 0) bitrate_ = bitrate;
    if (!encoder.empty()) encoder_name_ = encoder;
}

void RTMPSinkNode::setOutputSize(int width, int height) {
    output_width_ = width;
    output_height_ = height;
}

bool RTMPSinkNode::isConnected() const {
    return connected_;
}

bool RTMPSinkNode::onStartup() {
    if (output_url_.empty()) {
        output_url_ = "rtmp://localhost/live/out1";
        LOG_WARN_FMT("[RTMPSink] Output URL not set, using default: {}", output_url_);
    }

    // 未显式设置 output_size 时延迟到首帧，用输入视频宽高初始化
    if (output_width_ > 0 && output_height_ > 0) {
        if (!initEncoder(output_width_, output_height_)) {
            LOG_ERROR("[RTMPSink] Failed to initialize encoder");
            return false;
        }
    } else {
        LOG_INFO("[RTMPSink] Output size not set, will use first frame dimensions");
    }

    LOG_INFO_FMT("[RTMPSink] Started pushing to {}", output_url_);
    return true;
}

void RTMPSinkNode::onShutdown() {
    closeEncoder();
    LOG_INFO("[RTMPSink] Stopped");
}

void RTMPSinkNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (!packet) return;

    // STREAM_END：QueuedNode 已置 running_=false，此处只需广播给下游
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO("[RTMPSink] Stream end");
        broadcast(packet);
        return;
    }

    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    if (!frame || !frame->mat || frame->mat->empty()) return;

    // 延迟初始化：没有显式 output_size 时取首帧宽高
    if (!encoder_) {
        if (!initEncoder(frame->mat->cols, frame->mat->rows)) {
            LOG_ERROR("[RTMPSink] Failed to initialize encoder on first frame");
            return;
        }
    }

    if (!encoder_->encodeFrame(frame->mat->data, frame->mat->cols, frame->mat->rows,
                               frame->mat->step, next_pts_++)) {
        LOG_ERROR("[RTMPSink] Failed to encode frame");
        connected_ = false;
        return;
    }
    connected_ = true;
}

bool RTMPSinkNode::initEncoder(int width, int height) {
    int w = (output_width_ > 0 && output_height_ > 0) ? output_width_ : width;
    int h = (output_width_ > 0 && output_height_ > 0) ? output_height_ : height;
    if (w <= 0 || h <= 0) {
        LOG_ERROR_FMT("[RTMPSink] Invalid output size {}x{}", w, h);
        return false;
    }

    encoder_ = std::make_unique<RTMPEncoder>();
    return encoder_->init(output_url_, "flv", w, h, bitrate_, encoder_name_);
}

void RTMPSinkNode::closeEncoder() {
    connected_ = false;
    if (encoder_) {
        encoder_->close();
        encoder_.reset();
    }
}

REGISTER_NODE("rtmp_sink", RTMPSinkNode)

} // namespace nodes
} // namespace ai_stream
