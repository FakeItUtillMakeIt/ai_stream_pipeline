// src/nodes/evidence/ftp_uploader.cpp
#include "ftp_uploader.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <curl/curl.h>
#include <filesystem>
#include <fstream>

namespace ai_stream {
namespace nodes {

FtpUploader::FtpUploader() = default;

FtpUploader::~FtpUploader() {
    shutdown();
}

bool FtpUploader::initialize(const FtpConfig& config) {
    config_ = config;

    curl_global_init(CURL_GLOBAL_DEFAULT);
    running_ = true;
    stop_flag_ = false;
    worker_ = std::thread(&FtpUploader::uploadLoop, this);

    LOG_INFO_FMT("[FtpUploader] Initialized: server={}, port={}, remote_dir={}",
                 config_.server, config_.port, config_.remote_dir);
    return true;
}

void FtpUploader::shutdown() {
    if (!running_) return;

    stop_flag_ = true;
    queue_cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
    running_ = false;
    curl_global_cleanup();
    LOG_INFO("[FtpUploader] Shutdown");
}

void FtpUploader::enqueue(const std::string& local_path) {
    if (!running_ || local_path.empty()) return;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        upload_queue_.push(local_path);
    }
    queue_cv_.notify_one();
    LOG_INFO_FMT("[FtpUploader] Enqueued: {}", local_path);
}

size_t FtpUploader::queueSize() const {
    std::lock_guard<std::mutex> lock(queue_mutex_);
    return upload_queue_.size();
}

void FtpUploader::uploadLoop() {
    while (!stop_flag_) {
        std::string local_path;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !upload_queue_.empty() || stop_flag_; });
            if (stop_flag_ && upload_queue_.empty()) break;
            if (upload_queue_.empty()) continue;
            local_path = std::move(upload_queue_.front());
            upload_queue_.pop();
        }

        if (!local_path.empty()) {
            bool success = false;
            for (int retry = 0; retry <= config_.max_retry; ++retry) {
                if (retry > 0) {
                    LOG_WARN_FMT("[FtpUploader] Retry {}/{} for: {}", retry, config_.max_retry, local_path);
                    std::this_thread::sleep_for(std::chrono::seconds(2 * retry));
                }
                if (uploadFile(local_path)) {
                    success = true;
                    break;
                }
            }

            if (success) {
                uploaded_count_++;
                if (config_.delete_after_upload) {
                    std::error_code ec;
                    std::filesystem::remove(local_path, ec);
                    if (ec) {
                        LOG_WARN_FMT("[FtpUploader] Failed to delete local file: {}", local_path);
                    }
                }
            } else {
                failed_count_++;
                LOG_ERROR_FMT("[FtpUploader] Failed to upload after {} retries: {}", config_.max_retry, local_path);
            }
        }
    }
}

bool FtpUploader::uploadFile(const std::string& local_path) {
    CURL* curl = curl_easy_init();
    if (!curl) {
        LOG_ERROR("[FtpUploader] Failed to init curl");
        return false;
    }

    std::ifstream file(local_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR_FMT("[FtpUploader] Failed to open file: {}", local_path);
        curl_easy_cleanup(curl);
        return false;
    }

    auto file_size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::string remote_url = buildRemoteUrl(std::filesystem::path(local_path).filename().string());
    std::string userpwd = config_.username + ":" + config_.password;

    std::string response_data;
    curl_easy_setopt(curl, CURLOPT_URL, remote_url.c_str());
    curl_easy_setopt(curl, CURLOPT_USERNAME, config_.username.c_str());
    curl_easy_setopt(curl, CURLOPT_PASSWORD, config_.password.c_str());
    curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
    curl_easy_setopt(curl, CURLOPT_READDATA, &file);
    curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, static_cast<curl_off_t>(file_size));
    curl_easy_setopt(curl, CURLOPT_FTP_CREATE_MISSING_DIRS, CURLFTP_CREATE_DIR);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT, 60L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT, 15L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);

    if (config_.use_tls) {
        curl_easy_setopt(curl, CURLOPT_USE_SSL, CURLUSESSL_ALL);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, 0L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0L);
    }

    CURLcode res = curl_easy_perform(curl);

    long http_code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

    curl_easy_cleanup(curl);

    if (res != CURLE_OK) {
        LOG_ERROR_FMT("[FtpUploader] Upload failed: {} (curl error: {})",
                      local_path, curl_easy_strerror(res));
        return false;
    }

    if (http_code >= 400) {
        LOG_ERROR_FMT("[FtpUploader] Upload failed: {} (http code: {})", local_path, http_code);
        return false;
    }

    LOG_INFO_FMT("[FtpUploader] Uploaded successfully: {} -> {}", local_path, remote_url);
    return true;
}

std::string FtpUploader::buildRemoteUrl(const std::string& filename) const {
    std::string url = "ftp://";
    url += config_.server;
    if (config_.port != 21) {
        url += ":" + std::to_string(config_.port);
    }
    url += config_.remote_dir;
    if (!config_.remote_dir.empty() && config_.remote_dir.back() != '/') {
        url += "/";
    }
    url += filename;
    return url;
}

} // namespace nodes
} // namespace ai_stream
