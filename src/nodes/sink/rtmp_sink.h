// src/nodes/sink/rtmp_sink.h
#pragma once

#include "ai_stream/nodes/i_sink_node.h"
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>

struct AVFormatContext;
struct AVCodecContext;
struct AVFrame;
struct SwsContext;

namespace ai_stream {
namespace nodes {

class RTMPSinkNode : public ISinkNode {
public:
    RTMPSinkNode();
    ~RTMPSinkNode() override;

    // ISinkNode 接口
    void setTarget(const std::string& target) override;
    void setEncodingParams(int bitrate, const std::string& encoder) override;
    void setOutputSize(int width, int height) override;
    bool isConnected() const override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void encoderLoop();
    bool initEncoder();
    void closeEncoder();
    bool encodeAndSend(std::shared_ptr<core::VideoFramePacket> frame);

    std::string output_url_;
    int output_width_ = 0;
    int output_height_ = 0;
    int bitrate_ = 4000; // kbps
    std::string encoder_name_ = "libx264";

    std::atomic<bool> running_{false};
    std::thread worker_;
    
    // 待编码帧队列（有界阻塞）
    std::queue<std::shared_ptr<core::VideoFramePacket>> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    static constexpr size_t MAX_QUEUE_SIZE = 30;

    // FFmpeg 组件
    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVFrame* av_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    int64_t frame_pts_ = 0;
    std::atomic<bool> connected_{false};
};

} // namespace nodes
} // namespace ai_stream