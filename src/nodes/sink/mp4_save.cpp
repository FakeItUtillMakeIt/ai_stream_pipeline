// src/nodes/sink/mp4_save.cpp
#include "mp4_save.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace nodes {

MP4SaveNode::MP4SaveNode() : ISinkNode("MP4Save"),file_path_("file://./out.mp4") {}
MP4SaveNode::~MP4SaveNode() { stop(); }

void MP4SaveNode::setTarget(const std::string& target) {
    // target 格式 "file:///path/to/output.mp4"
    file_path_ = target;
    LOG_INFO_FMT("mp4 save path:{}",file_path_);
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
        LOG_ERROR_FMT("[MP4Save] Output file path not set");
        return false;
    }
    if (!initFileWriter()) return false;
    running_ = true;
    worker_ = std::thread(&MP4SaveNode::writerLoop, this);
    LOG_INFO_FMT("[MP4Save] Started recording to {}", file_path_);
    return true;
}

void MP4SaveNode::stop() {
    running_ = false;
    queue_cv_.notify_all();
    if (worker_.joinable()) worker_.join();
    closeFileWriter();
}

void MP4SaveNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (!running_) return;
    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (frame_queue_.size() >= MAX_QUEUE_SIZE) {
            frame_queue_.pop(); // 丢弃最老的
        }
        frame_queue_.push(frame);
    }
    queue_cv_.notify_one();
}

void MP4SaveNode::writerLoop() {
    while (running_) {
        std::shared_ptr<core::VideoFramePacket> frame;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !frame_queue_.empty() || !running_; });
            if (!running_) break;
            frame = frame_queue_.front();
            frame_queue_.pop();
        }
        
        // 编码并写入文件 
        

    }
}

bool MP4SaveNode::initFileWriter() {
    // 实际实现：avformat_alloc_output_context2, 添加视频流, 打开文件等
    return true;
}

void MP4SaveNode::closeFileWriter() {
    // 写入文件尾，释放资源
}

REGISTER_NODE("mp4_save", MP4SaveNode)

} // namespace nodes
} // namespace ai_stream