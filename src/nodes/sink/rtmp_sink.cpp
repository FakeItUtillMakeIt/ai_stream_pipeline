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

RTMPSinkNode::~RTMPSinkNode() = default;

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

    if (!initEncoder()) {
        LOG_ERROR("[RTMPSink] Failed to initialize encoder");
        return false;
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

    int width = output_width_ > 0 ? output_width_ : frame->width;
    int height = output_height_ > 0 ? output_height_ : frame->height;

    if (!encoder_->encodeFrame(frame->mat->data, width, height,
                                frame->mat->step, next_pts_++)) {
        LOG_ERROR("[RTMPSink] Failed to encode frame");
        connected_ = false;
        return;
    }
    connected_ = true;
}

bool RTMPSinkNode::initEncoder() {
    encoder_ = std::make_unique<RTMPEncoder>();
    return encoder_->init(output_url_, "flv",
                          output_width_ > 0 ? output_width_ : 1920,
                          output_height_ > 0 ? output_height_ : 1080,
                          bitrate_, encoder_name_);
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
