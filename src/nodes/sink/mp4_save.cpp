// src/nodes/sink/mp4_save.cpp
#include "mp4_save.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "encoder_base.h"
#include <filesystem>
#include <chrono>

namespace ai_stream {
namespace nodes {

MP4SaveNode::MP4SaveNode() : ISinkNode("MP4Save") {
    LOG_DEBUG("[MP4Save] Constructor");
}

MP4SaveNode::~MP4SaveNode() { 
    stop(); 
    LOG_DEBUG("[MP4Save] Destructor");
}

void MP4SaveNode::setTarget(const std::string& target) {
    file_path_ = target;
    LOG_INFO_FMT("[MP4Save] Target: {}", file_path_);
    if (file_path_.find("file://") == 0) {
        file_path_ = file_path_.substr(7);
    }
}

void MP4SaveNode::setEncodingParams(int bitrate, const std::string& encoder) {
    bitrate_ = bitrate;
    encoder_name_ = encoder;
}

void MP4SaveNode::setOutputSize(int width, int height) {
    output_width_ = width;
    output_height_ = height;
}

bool MP4SaveNode::start() {
    if (file_path_.empty()) {
        // 生成默认路径
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        std::stringstream ss;
        ss << "./output/video_" << time_t << ".mp4";
        file_path_ = ss.str();
        LOG_WARN_FMT("[MP4Save] Output path not set, using default: {}", file_path_);
    }
    
    // 确保目录存在
    std::filesystem::path file_path(file_path_);
    auto parent = file_path.parent_path();
    if (!parent.empty() && !std::filesystem::exists(parent)) {
        std::error_code ec;
        std::filesystem::create_directories(parent, ec);
        if (ec) {
            LOG_ERROR_FMT("[MP4Save] Failed to create directory: {} ({})", 
                          parent.string(), ec.message());
            return false;
        }
    }
    
    // 确保 .mp4 扩展名
    std::string path_str = file_path.string();
    if (path_str.find(".mp4") == std::string::npos) {
        path_str += ".mp4";
    }
    
    if (!initFileWriter()) {
        LOG_ERROR("[MP4Save] Failed to initialize file writer");
        return false;
    }
    
    running_ = true;
    next_pts_ = 0;
    frame_queue_.reset();
    worker_ = std::thread(&MP4SaveNode::writerLoop, this);
    LOG_INFO_FMT("[MP4Save] Started recording to {}", path_str);
    return true;
}

void MP4SaveNode::stop() {
    if (!running_) return;
    
    running_ = false;
    frame_queue_.stop();
    
    if (worker_.joinable()) {
        worker_.join();
    }
    
    closeFileWriter();
    LOG_INFO("[MP4Save] Stopped");
}

void MP4SaveNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[MP4Save] Received stream end");
        stop();
        broadcast(packet);
        return;
    }
    if (!running_) return;
    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    if (!frame || !frame->mat || frame->mat->empty()) return;

    // 队列满时丢弃最旧帧
    while (!frame_queue_.tryPush(frame)) {
        std::shared_ptr<core::VideoFramePacket> discarded;
        if (!frame_queue_.tryPop(discarded)) break;
        LOG_WARN_FMT("[MP4Save] Queue full, dropping oldest frame");
    }
}

void MP4SaveNode::writerLoop() {
    while (running_) {
        std::shared_ptr<core::VideoFramePacket> frame;
        if (!frame_queue_.pop(frame, std::chrono::milliseconds(100))) continue;

        if (!frame || !frame->mat || frame->mat->empty()) continue;
        
        encoder_->encodeFrame(frame->mat->data, 
                              frame->width, frame->height, 
                              frame->mat->step, next_pts_++);
    }
    
}

bool MP4SaveNode::initFileWriter() {
    int w = output_width_ > 0 ? output_width_ : 1920;
    int h = output_height_ > 0 ? output_height_ : 1080;
    
    encoder_ = std::make_unique<FileEncoder>();
    return encoder_->init(file_path_, "mp4", w, h, bitrate_, encoder_name_);
}

void MP4SaveNode::closeFileWriter() {
    if (encoder_) {
        encoder_->close();
        encoder_.reset();
    }
}

REGISTER_NODE("mp4_save", MP4SaveNode)

} // namespace nodes
} // namespace ai_stream