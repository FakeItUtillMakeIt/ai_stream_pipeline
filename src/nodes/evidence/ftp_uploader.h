// src/nodes/evidence/ftp_uploader.h
#pragma once

#include "ai_stream/core/bounded_queue.h"
#include <string>
#include <thread>
#include <mutex>
#include <atomic>

namespace ai_stream {
namespace nodes {

struct FtpConfig {
    std::string server;
    int port = 21;
    std::string username;
    std::string password;
    std::string remote_dir = "/evidence";
    bool use_tls = false;
    bool delete_after_upload = false;
    int max_retry = 3;
};

class FtpUploader {
public:
    FtpUploader();
    ~FtpUploader();

    bool initialize(const FtpConfig& config);
    void shutdown();
    void enqueue(const std::string& local_path);
    size_t queueSize() const;
    size_t uploadedCount() const { return uploaded_count_; }
    size_t failedCount() const { return failed_count_; }

private:
    void uploadLoop();
    bool uploadFile(const std::string& local_path);
    std::string buildRemoteUrl(const std::string& filename) const;

    FtpConfig config_;
    std::atomic<bool> running_{false};
    std::atomic<bool> stop_flag_{false};

    // 上传任务队列（证据文件不允许丢弃，满时阻塞等待）
    core::BoundedQueue<std::string> upload_queue_{1024};
    std::thread worker_;

    size_t uploaded_count_ = 0;
    size_t failed_count_ = 0;
    mutable std::mutex mutex_;
};

} // namespace nodes
} // namespace ai_stream
