// src/nodes/evidence/video_recorder.h
#pragma once

#include "ai_stream/core/packet.h"
#include "sink/encoder_base.h"
#include <memory>
#include <string>
#include <deque>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>

namespace ai_stream {
namespace nodes {

class VideoRecorder {
public:
    VideoRecorder();
    ~VideoRecorder();

    bool initialize(const std::string& output_dir, int fps, int bitrate);
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

    std::atomic<bool> recording_{false};
    std::atomic<bool> stop_flag_{false};
    std::string current_filepath_;
    mutable std::mutex mutex_;

    std::unique_ptr<FileEncoder> encoder_;
    int64_t pts_ = 0;

    std::queue<std::shared_ptr<core::VideoFramePacket>> frame_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread encode_thread_;
};

} // namespace nodes
} // namespace ai_stream
