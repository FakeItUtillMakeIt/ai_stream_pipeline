// src/nodes/sink/rtmp_sink.h
#pragma once

#include "ai_stream/nodes/i_sink_node.h"
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace ai_stream {
namespace nodes {

class EncoderBase;

class RTMPSinkNode : public ISinkNode {
public:
    RTMPSinkNode();
    ~RTMPSinkNode() override;

    void setTarget(const std::string& target) override;
    void setEncodingParams(int bitrate, const std::string& encoder) override;
    void setOutputSize(int width, int height) override;
    bool isConnected() const override;

    bool start() override;
    void stop() override;
    bool isRunning() const override{return running_.load();}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void encoderLoop();
    bool initEncoder();
    void closeEncoder();

    std::string output_url_;
    int output_width_ = 0;
    int output_height_ = 0;
    int bitrate_ = 4000;
    std::string encoder_name_ = "libx264";

    std::atomic<bool> running_{false};
    std::thread worker_;
    
    std::queue<std::shared_ptr<core::VideoFramePacket>> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    static constexpr size_t MAX_QUEUE_SIZE = 30;

    std::unique_ptr<EncoderBase> encoder_;
    int64_t next_pts_ = 0;
    std::atomic<bool> connected_{false};
};

} // namespace nodes
} // namespace ai_stream