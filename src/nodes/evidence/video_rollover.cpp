// src/nodes/evidence/video_rollover.cpp
#include "video_rollover.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <filesystem>

namespace ai_stream {
namespace nodes {

VideoRollover::VideoRollover() = default;

VideoRollover::~VideoRollover() {
    stop();
}

void VideoRollover::configure(const VideoRolloverConfig& config) {
    config_ = config;
    LOG_INFO_FMT("[VideoRollover] Configured: dir={}, retention={}h, interval={}min",
                 config_.video_dir, config_.retention_hours, config_.check_interval_min);
}

void VideoRollover::start() {
    if (!config_.enabled) {
        LOG_INFO("[VideoRollover] Disabled, not starting");
        return;
    }

    if (running_) return;

    running_ = true;
    stop_flag_ = false;
    worker_ = std::thread(&VideoRollover::rolloverLoop, this);
    LOG_INFO("[VideoRollover] Started");
}

void VideoRollover::stop() {
    if (!running_.exchange(false)) return;

    stop_flag_ = true;
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    LOG_INFO("[VideoRollover] Stopped");
}

void VideoRollover::forceCheck() {
    cleanupExpiredFiles();
}

void VideoRollover::rolloverLoop() {
    while (!stop_flag_) {
        cleanupExpiredFiles();

        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait_for(lock, std::chrono::minutes(config_.check_interval_min),
                     [this] { return stop_flag_.load(); });
    }
}

void VideoRollover::cleanupExpiredFiles() {
    if (!std::filesystem::exists(config_.video_dir)) {
        return;
    }

    auto now = std::filesystem::file_time_type::clock::now();
    auto retention = std::filesystem::file_time_type::duration::zero();
    retention = std::chrono::duration_cast<std::filesystem::file_time_type::duration>(
        std::chrono::hours(config_.retention_hours));

    size_t deleted = 0;
    std::error_code ec;

    for (auto& entry : std::filesystem::directory_iterator(config_.video_dir, ec)) {
        if (ec) break;
        if (!entry.is_regular_file()) continue;

        auto last_write = entry.last_write_time(ec);
        if (ec) continue;

        if (now - last_write > retention) {
            std::filesystem::remove(entry.path(), ec);
            if (!ec) {
                deleted++;
                LOG_DEBUG_FMT("[VideoRollover] Deleted expired file: {}", entry.path().string());
            }
        }
    }

    if (deleted > 0) {
        deleted_count_ += deleted;
        LOG_INFO_FMT("[VideoRollover] Cleaned up {} expired files (total: {})", deleted, deleted_count_);
    }
}

} // namespace nodes
} // namespace ai_stream
