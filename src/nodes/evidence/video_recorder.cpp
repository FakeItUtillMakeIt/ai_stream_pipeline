// src/nodes/evidence/video_recorder.cpp
#include "video_recorder.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <chrono>
#include <filesystem>

namespace ai_stream {
namespace nodes {

VideoRecorder::VideoRecorder() = default;

VideoRecorder::~VideoRecorder() {
    shutdown();
}

bool VideoRecorder::initialize(const std::string& output_dir, int fps, int bitrate, const std::string& codec) {
    // pipeline 重启场景：先释放上一次的线程/编码器
    shutdown();

    output_dir_ = output_dir;
    fps_ = fps;
    bitrate_ = bitrate;
    codec_ = codec;

    std::error_code ec;
    std::filesystem::create_directories(output_dir_, ec);
    if (ec) {
        LOG_ERROR_FMT("[VideoRecorder] Failed to create output directory: {} ({})",
                      output_dir_, ec.message());
        return false;
    }

    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = false;
        stop_requested_ = false;
        start_pending_ = false;
        start_ack_ = false;
        start_ok_ = false;
        worker_done_ = true;
        recording_ = false;
        encode_thread_ = std::thread(&VideoRecorder::encodingLoop, this);
    }

    LOG_INFO_FMT("[VideoRecorder] Initialized: dir={}, fps={}, bitrate={}, codec={}",
                 output_dir_, fps_, bitrate_, codec_);
    return true;
}

bool VideoRecorder::startRecording(
    const std::string& filename,
    std::deque<std::shared_ptr<core::VideoFramePacket>> pre_frames) {

    if (pre_frames.empty()) {
        LOG_ERROR("[VideoRecorder] Cannot start recording with empty pre-frames");
        return false;
    }
    const auto& first_frame = pre_frames.front();
    if (!first_frame || !first_frame->mat || first_frame->mat->empty()) {
        LOG_ERROR("[VideoRecorder] Invalid first pre-frame");
        return false;
    }

    const int width = first_frame->mat->cols;
    const int height = first_frame->mat->rows;

    std::filesystem::path filepath = output_dir_;
    filepath /= filename;

    {
        std::unique_lock<std::mutex> lock(mutex_);

        if (!encode_thread_.joinable() || shutdown_) {
            LOG_ERROR("[VideoRecorder] Encoder thread not running");
            return false;
        }
        if (!worker_done_ || recording_) {
            LOG_WARN("[VideoRecorder] Already recording, ignoring new request");
            return false;
        }

        // 先把 pre-frames 入队，worker 启动后按顺序编码；实时帧自然排在其后
        frame_queue_.reset();
        for (const auto& frame : pre_frames) {
            if (frame && frame->mat && !frame->mat->empty()) {
                frame_queue_.tryPush(frame);
            }
        }

        pending_filepath_ = filepath.string();
        pending_width_ = width;
        pending_height_ = height;
        current_filepath_ = pending_filepath_;

        start_ack_ = false;
        start_ok_ = false;
        stop_requested_ = false;
        worker_done_ = false;
        start_pending_ = true;

        cv_.notify_all();
        cv_.wait(lock, [this] { return start_ack_ || shutdown_; });

        if (shutdown_ || !start_ack_ || !start_ok_) {
            recording_ = false;
            worker_done_ = true;
            frame_queue_.clear();
            LOG_ERROR_FMT("[VideoRecorder] Failed to start encoder for: {}", current_filepath_);
            return false;
        }

        recording_ = true;
    }

    LOG_INFO_FMT("[VideoRecorder] Started recording: {} ({} pre-frames written)",
                 filepath.string(), pre_frames.size());
    return true;
}

void VideoRecorder::enqueueFrame(std::shared_ptr<core::VideoFramePacket> frame) {
    if (!recording_.load(std::memory_order_relaxed) || !frame || !frame->mat || frame->mat->empty()) return;

    // 队列满时丢弃最旧帧
    while (!frame_queue_.tryPush(frame)) {
        std::shared_ptr<core::VideoFramePacket> discarded;
        if (!frame_queue_.tryPop(discarded)) break;
    }
}

void VideoRecorder::stop() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (worker_done_) {
        recording_ = false;
        return;
    }

    stop_requested_ = true;
    cv_.notify_all();
    cv_.wait(lock, [this] { return worker_done_; });
    recording_ = false;
}

void VideoRecorder::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        shutdown_ = true;
        stop_requested_ = true;
        frame_queue_.stop();
        cv_.notify_all();
    }

    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }

    std::lock_guard<std::mutex> lock(mutex_);
    shutdown_ = false;
    stop_requested_ = false;
    start_pending_ = false;
    start_ack_ = false;
    start_ok_ = false;
    worker_done_ = true;
    recording_ = false;
    pending_filepath_.clear();
    current_filepath_.clear();
}

bool VideoRecorder::isRecording() const {
    return recording_.load();
}

std::string VideoRecorder::getCurrentFilePath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_filepath_;
}

void VideoRecorder::encodingLoop() {
    ScopedLogContext log_ctx(log_pipeline_, log_node_);
    std::unique_lock<std::mutex> lock(mutex_);

    while (!shutdown_) {
        cv_.wait(lock, [this] { return shutdown_ || start_pending_; });
        if (shutdown_) break;

        start_pending_ = false;
        const std::string filepath = pending_filepath_;
        const int width = pending_width_;
        const int height = pending_height_;

        // 编码器初始化可能较慢（首次 NVENC/CUDA），不持锁执行
        lock.unlock();
        const bool ok = startEncoder(filepath, width, height);
        lock.lock();

        start_ok_ = ok;
        start_ack_ = true;
        if (!ok) {
            recording_ = false;
            worker_done_ = true;
            stop_requested_ = false;
            cv_.notify_all();
            continue;
        }
        cv_.notify_all();
        lock.unlock();

        // 录制循环：排队帧消费完后，收到 stop 请求即收尾
        while (true) {
            std::shared_ptr<core::VideoFramePacket> frame;
            const bool got = frame_queue_.pop(frame, std::chrono::milliseconds(100));
            if (got) {
                if (frame && frame->mat && !frame->mat->empty() &&
                    frame->mat->cols == width_ && frame->mat->rows == height_) {
                    // 统一以初始化尺寸编码；尺寸不一致的帧跳过，避免破坏码流
                    encoder_->encodeFrame(frame->mat->data, width_, height_,
                                          static_cast<int>(frame->mat->step), pts_++);
                }
                continue;
            }

            std::lock_guard<std::mutex> lk(mutex_);
            if (shutdown_) break;
            if (stop_requested_ && frame_queue_.empty()) break;
        }

        finishEncoder();

        lock.lock();
        recording_ = false;
        worker_done_ = true;
        stop_requested_ = false;
        cv_.notify_all();
    }

    lock.unlock();
    if (encoder_) {
        encoder_->close();
        encoder_.reset();
    }
}

bool VideoRecorder::startEncoder(const std::string& filepath, int width, int height) {
    // 输入尺寸变化时重建编码器；尺寸稳定时复用编码器，仅切换 MP4 输出
    if (encoder_ && (width != width_ || height != height_)) {
        LOG_INFO_FMT("[VideoRecorder] Input size changed {}x{} -> {}x{}, recreating encoder",
                     width_, height_, width, height);
        closeEncoder();
    }

    if (!encoder_) {
        auto enc = std::make_unique<FileEncoder>();
        if (!enc->init(filepath, "mp4", width, height, bitrate_, codec_)) {
            LOG_ERROR_FMT("[VideoRecorder] Failed to initialize encoder for: {}", filepath);
            return false;
        }
        encoder_ = std::move(enc);
        width_ = width;
        height_ = height;
        pts_ = 0;
        return true;
    }

    if (!encoder_->startFile(filepath, "mp4")) {
        LOG_WARN_FMT("[VideoRecorder] Failed to reuse encoder for: {}, recreating", filepath);
        closeEncoder();

        auto enc = std::make_unique<FileEncoder>();
        if (!enc->init(filepath, "mp4", width, height, bitrate_, codec_)) {
            LOG_ERROR_FMT("[VideoRecorder] Failed to initialize encoder for: {}", filepath);
            return false;
        }
        encoder_ = std::move(enc);
        width_ = width;
        height_ = height;
        pts_ = 0;
    }
    return true;
}

void VideoRecorder::finishEncoder() {
    if (encoder_) {
        encoder_->finishFile();
    }
    LOG_INFO_FMT("[VideoRecorder] Stopped recording: {}", current_filepath_);
}

void VideoRecorder::closeEncoder() {
    if (encoder_) {
        encoder_->close();
        encoder_.reset();
    }
    width_ = 0;
    height_ = 0;
}

} // namespace nodes
} // namespace ai_stream
