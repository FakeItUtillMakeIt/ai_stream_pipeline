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
    if (config_.check_interval_min == 0) config_.check_interval_min = 10;
    LOG_INFO_FMT("[VideoRollover] Configured: dir={}, snapshot_dir={}, retention={}h, interval={}min",
                 config_.video_dir,
                 config_.snapshot_dir.empty() ? std::string("-") : config_.snapshot_dir,
                 config_.retention_hours, config_.check_interval_min);
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
    ScopedLogContext log_ctx(log_pipeline_, log_node_);
    // 启动时先检查并清理一次过期文件，之后按周期清理
    cleanupExpiredFiles();
    while (!stop_flag_) {
        {
            std::unique_lock<std::mutex> lock(mutex_);
            cv_.wait_for(lock, std::chrono::minutes(config_.check_interval_min),
                         [this] { return stop_flag_.load(); });
        }
        if (stop_flag_) break;
        cleanupExpiredFiles();
    }
}

void VideoRollover::cleanupExpiredFiles() {
    auto now = std::filesystem::file_time_type::clock::now();
    const auto retention = std::chrono::duration_cast<std::filesystem::file_time_type::duration>(
        std::chrono::hours(config_.retention_hours));

    size_t deleted = 0;
    std::error_code ec;

    // 视频与快照共用同一保留策略；空目录/重复目录跳过
    const std::string dirs[2] = {config_.video_dir, config_.snapshot_dir};
    for (int i = 0; i < 2; ++i) {
        const std::string& dir = dirs[i];
        if (dir.empty()) continue;
        if (i == 1 && dir == dirs[0]) continue;
        if (!std::filesystem::exists(dir, ec)) continue;

        for (auto& entry : std::filesystem::directory_iterator(dir, ec)) {
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
        ec.clear();
    }

    if (deleted > 0) {
        deleted_count_ += deleted;
        LOG_INFO_FMT("[VideoRollover] Cleaned up {} expired files (total: {})", deleted, deleted_count_);
    }
}

} // namespace nodes
} // namespace ai_stream
