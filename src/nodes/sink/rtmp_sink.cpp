// src/nodes/sink/rtmp_sink.cpp
#include "rtmp_sink.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "encoder_base.h"
#include "mpp_encoder.h"
#include <opencv2/opencv.hpp>

namespace ai_stream {
namespace nodes {

RTMPSinkNode::RTMPSinkNode() : ISinkNode("RTMPSink") {
    LOG_DEBUG("[RTMPSink] Constructor");
}

RTMPSinkNode::~RTMPSinkNode() { 
    stop(); 
    LOG_DEBUG("[RTMPSink] Destructor");
}

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

bool RTMPSinkNode::start() {
    if (output_url_.empty()) {
        output_url_ = "rtmp://localhost/live/out1";
        LOG_WARN_FMT("[RTMPSink] Output URL not set, using default: {}", output_url_);
    }

    if (!initEncoder()) {
        LOG_ERROR("[RTMPSink] Failed to initialize encoder");
        return false;
    }

    // 如果之前的 worker 线程还未 join，先 join 它（自停后线程可能还在运行）
    if (worker_.joinable()) {
        worker_.join();
    }

    running_ = true;
    frame_queue_.reset();
    worker_ = std::thread(&RTMPSinkNode::encoderLoop, this);
    LOG_INFO_FMT("[RTMPSink] Started pushing to {}", output_url_);
    return true;
}

void RTMPSinkNode::stop() {
    if (!running_) return;
    
    running_ = false;
    frame_queue_.stop();
    
    if (worker_.joinable()) {
        worker_.join();
    }
    
    closeEncoder();
    LOG_INFO("[RTMPSink] Stopped");
}

void RTMPSinkNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[RTMPSink] Received stream end");
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 encoderLoop 中检查，worker 线程会自然退出
        broadcast(packet);
        return;
    }
    if (!running_) return;
    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    if (!frame || !frame->mat || frame->mat->empty()) return;

    // 队列满时丢弃最旧帧，保证直播实时性
    while (!frame_queue_.tryPush(frame)) {
        std::shared_ptr<core::VideoFramePacket> discarded;
        if (!frame_queue_.tryPop(discarded)) break;
        LOG_WARN_FMT("[RTMPSink] Queue full, dropping oldest frame");
    }
}

void RTMPSinkNode::encoderLoop() {
    while (running_) {
        std::shared_ptr<core::VideoFramePacket> frame;
        if (!frame_queue_.pop(frame, std::chrono::milliseconds(100))) continue;

        // 检查是否为流结束信号
        if (frame->type == core::PacketType::STREAM_END) {
            LOG_INFO_FMT("[RTMPSink] Stream end received in worker thread");
            break;
        }

        if (!frame || !frame->mat || frame->mat->empty()) continue;

        int width = output_width_ > 0 ? output_width_ : frame->width;
        int height = output_height_ > 0 ? output_height_ : frame->height;

        if (!encoder_->encodeFrame(frame->mat->data, width, height,
                                    frame->mat->step, next_pts_++)) {
            LOG_ERROR_FMT("[RTMPSink] Failed to encode frame");
            connected_ = false;
            break;
        }
        connected_ = true;
    }

}

bool RTMPSinkNode::initEncoder() {
    // MPP 硬编码（RK 平台）：encoder_name 含 "mpp" 时尝试，失败回退软编
    if (encoder_name_.find("mpp") != std::string::npos) {
        auto mpp = std::make_unique<MppEncoder>();
        if (mpp->init(output_url_, "flv",
                      output_width_ > 0 ? output_width_ : 1920,
                      output_height_ > 0 ? output_height_ : 1080,
                      bitrate_, encoder_name_)) {
            encoder_ = std::move(mpp);
            LOG_INFO("[RTMPSink] Using MPP hardware encoder");
            return true;
        }
        LOG_WARN("[RTMPSink] MPP encoder unavailable, falling back to software encoder");
    }
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