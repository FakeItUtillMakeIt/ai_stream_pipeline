// src/nodes/evidence/video_rollover.h
#pragma once

#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <chrono>

namespace ai_stream {
namespace nodes {

struct VideoRolloverConfig {
    std::string video_dir = "./evidence/videos";
    std::string snapshot_dir;   // 可选：与视频共用同一保留策略
    uint32_t retention_hours = 24;
    uint32_t check_interval_min = 10;
    bool enabled = true;
};

class VideoRollover {
public:
    VideoRollover();
    ~VideoRollover();

    void configure(const VideoRolloverConfig& config);
    void start();
    void stop();

    void setLogContext(const std::string& pipeline, const std::string& node) {
        log_pipeline_ = pipeline;
        log_node_ = node;
    }
    void forceCheck();
    size_t getDeletedCount() const { return deleted_count_; }

private:
    void rolloverLoop();
    void cleanupExpiredFiles();

    VideoRolloverConfig config_;
    std::string log_pipeline_;
    std::string log_node_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_flag_{false};
    std::thread worker_;
    std::mutex mutex_;
    std::condition_variable cv_;
    size_t deleted_count_ = 0;
};

} // namespace nodes
} // namespace ai_stream
