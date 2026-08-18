// src/nodes/sink/mp4_save.h
#pragma once

#include "ai_stream/nodes/i_sink_node.h"
#include "ai_stream/core/bounded_queue.h"
#include <thread>
#include <atomic>
#include <memory>
#include <filesystem>
#include <sstream>

namespace ai_stream {
namespace nodes {

class EncoderBase;

class MP4SaveNode : public ISinkNode {
public:
    MP4SaveNode();
    ~MP4SaveNode() override;

    void setTarget(const std::string& target) override;
    void setEncodingParams(int bitrate, const std::string& encoder) override;
    void setOutputSize(int width, int height) override;
    bool isConnected() const override { return true; }

    bool start() override;
    void stop() override;
    bool isRunning() const override{return running_.load();}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void writerLoop();
    bool initFileWriter();
    void closeFileWriter();

    std::string file_path_;
    int output_width_ = 0;
    int output_height_ = 0;
    int bitrate_ = 4000;
    std::string encoder_name_ = "libx264";

    std::atomic<bool> running_{false};
    std::thread worker_;

    // 有界队列（满时丢最旧帧）
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> frame_queue_{100};

    std::unique_ptr<EncoderBase> encoder_;
    int64_t next_pts_ = 0;
};

} // namespace nodes
} // namespace ai_stream