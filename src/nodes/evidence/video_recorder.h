// src/nodes/evidence/video_recorder.h
#pragma once

#include "ai_stream/core/packet.h"
#include "ai_stream/core/bounded_queue.h"
#include "sink/encoder_base.h"
#include <memory>
#include <string>
#include <deque>
#include <atomic>
#include <thread>
#include <mutex>

namespace ai_stream {
namespace nodes {

class VideoRecorder {
public:
    VideoRecorder();
    ~VideoRecorder();

    bool initialize(const std::string& output_dir, int fps, int bitrate, const std::string& codec);
    bool startRecording(const std::string& filename,
                       std::deque<std::shared_ptr<core::VideoFramePacket>> pre_frames);
    void enqueueFrame(std::shared_ptr<core::VideoFramePacket> frame);
    void stop();
    bool isRecording() const;
    std::string getCurrentFilePath() const;

private:
    void encodingLoop();
    bool initEncoder(const std::string& filepath, int width, int height);
    void closeEncoder();

    std::string output_dir_;
    int fps_ = 25;
    int bitrate_ = 4000;
    std::string codec_ = "libx264";

    std::atomic<bool> recording_{false};
    std::atomic<bool> stop_flag_{false};
    std::string current_filepath_;
    mutable std::mutex mutex_;

    std::unique_ptr<FileEncoder> encoder_;
    int64_t pts_ = 0;

    // 有界队列（满时丢最旧帧，防止编码慢时无界增长）
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> frame_queue_{256};
    std::thread encode_thread_;
};

} // namespace nodes
} // namespace ai_stream
