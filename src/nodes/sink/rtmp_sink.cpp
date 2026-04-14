// src/nodes/sink/rtmp_sink.cpp
#include "rtmp_sink.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/opencv.hpp>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
}

namespace ai_stream {
namespace nodes {

RTMPSinkNode::RTMPSinkNode() : ISinkNode("RTMPSink"),output_url_("rtmp://localhost/live/out1") {}
RTMPSinkNode::~RTMPSinkNode() { stop(); }

void RTMPSinkNode::setTarget(const std::string& target) { output_url_ = target; }
void RTMPSinkNode::setEncodingParams(int bitrate, const std::string& encoder) {
    bitrate_ = bitrate;
    encoder_name_ = encoder;
}
void RTMPSinkNode::setOutputSize(int width, int height) {
    output_width_ = width;
    output_height_ = height;
}
bool RTMPSinkNode::isConnected() const { return connected_; }

bool RTMPSinkNode::start() {
    if (output_url_.empty()) {
        LOG_ERROR_FMT("[RTMPSink] Output URL not set");
        return false;
    }
    if (!initEncoder()) {
        return false;
    }
    running_ = true;
    worker_ = std::thread(&RTMPSinkNode::encoderLoop, this);
    LOG_INFO_FMT("[RTMPSink] Started pushing to {}", output_url_);
    return true;
}

void RTMPSinkNode::stop() {
    running_ = false;
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    closeEncoder();
}

void RTMPSinkNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (!running_) return;
    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (frame_queue_.size() >= MAX_QUEUE_SIZE) {
            // 队列满，丢弃最老的一帧
            frame_queue_.pop();
            LOG_WARN_FMT("[RTMPSink] Queue full, dropping oldest frame");
        }
        frame_queue_.push(frame);
    }
    queue_cv_.notify_one();
}

void RTMPSinkNode::encoderLoop() {
    while (running_) {
        std::shared_ptr<core::VideoFramePacket> frame;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !frame_queue_.empty() || !running_; });
            if (!running_) break;
            frame = frame_queue_.front();
            frame_queue_.pop();
        }
        
        if (!encodeAndSend(frame)) {
            LOG_ERROR_FMT("[RTMPSink] Failed to encode/send frame");
            // 可尝试重连
        }
    }
    
    // 冲刷编码器
    // avcodec_send_frame(codec_ctx_, nullptr);
}

bool RTMPSinkNode::initEncoder() {
    // 实际实现：分配 AVFormatContext, AVCodecContext, 打开输出 URL, 写入头信息
    // 此处简化为成功
    connected_ = true;
    return true;
}

void RTMPSinkNode::closeEncoder() {
    // 释放 FFmpeg 资源
    connected_ = false;
}

bool RTMPSinkNode::encodeAndSend(std::shared_ptr<core::VideoFramePacket> frame) {
    if (!frame->mat || frame->mat->empty()) return false;

    // 1. 将 cv::Mat 转换为 AVFrame
    // 2. 如果需要缩放，使用 sws_scale
    // 3. avcodec_send_frame / avcodec_receive_packet
    // 4. av_interleaved_write_frame
    
    // 模拟成功
    return true;
}

REGISTER_NODE("rtmp_sink", RTMPSinkNode)

} // namespace nodes
} // namespace ai_stream