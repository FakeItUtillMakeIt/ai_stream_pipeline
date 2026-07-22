// src/nodes/evidence/ftp_uploader.h
#pragma once

#include <string>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
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

    std::queue<std::string> upload_queue_;
    mutable std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::thread worker_;

    size_t uploaded_count_ = 0;
    size_t failed_count_ = 0;
    mutable std::mutex mutex_;
};

} // namespace nodes
} // namespace ai_stream
