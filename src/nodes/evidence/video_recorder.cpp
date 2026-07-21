// src/nodes/evidence/video_recorder.cpp
#include "video_recorder.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <filesystem>

namespace ai_stream {
namespace nodes {

VideoRecorder::VideoRecorder() = default;

VideoRecorder::~VideoRecorder() {
    stop_flag_ = true;
    recording_ = false;
    queue_cv_.notify_all();
    if (encode_thread_.joinable()) {
        encode_thread_.join();
    }
    closeEncoder();
}

bool VideoRecorder::initialize(const std::string& output_dir, int fps, int bitrate) {
    output_dir_ = output_dir;
    fps_ = fps;
    bitrate_ = bitrate;

    std::error_code ec;
    std::filesystem::create_directories(output_dir_, ec);
    if (ec) {
        LOG_ERROR_FMT("[VideoRecorder] Failed to create output directory: {} ({})",
                      output_dir_, ec.message());
        return false;
    }

    LOG_INFO_FMT("[VideoRecorder] Initialized: dir={}, fps={}, bitrate={}",
                 output_dir_, fps_, bitrate_);
    return true;
}

bool VideoRecorder::startRecording(
    const std::string& filename,
    std::deque<std::shared_ptr<core::VideoFramePacket>> pre_frames) {
    
    std::lock_guard<std::mutex> lock(mutex_);

    if (recording_) {
        LOG_WARN("[VideoRecorder] Already recording, ignoring new request");
        return false;
    }

    if (pre_frames.empty()) {
        LOG_ERROR("[VideoRecorder] Cannot start recording with empty pre-frames");
        return false;
    }

    auto first_frame = pre_frames.front();
    int width = first_frame->mat->cols;
    int height = first_frame->mat->rows;

    std::filesystem::path filepath = output_dir_;
    filepath /= filename;

    current_filepath_ = filepath.string();

    if (!initEncoder(current_filepath_, width, height)) {
        LOG_ERROR_FMT("[VideoRecorder] Failed to initialize encoder for: {}", current_filepath_);
        return false;
    }

    recording_ = true;
    stop_flag_ = false;
    pts_ = 0;

    encode_thread_ = std::thread(&VideoRecorder::encodingLoop, this);

    for (auto& frame : pre_frames) {
        if (frame && frame->mat && !frame->mat->empty()) {
            encoder_->encodeFrame(frame->mat->data, frame->width, frame->height,
                                  static_cast<int>(frame->mat->step), pts_++);
        }
    }

    LOG_INFO_FMT("[VideoRecorder] Started recording: {} ({} pre-frames written)",
                 current_filepath_, pre_frames.size());
    return true;
}

void VideoRecorder::enqueueFrame(std::shared_ptr<core::VideoFramePacket> frame) {
    if (!recording_ || !frame || !frame->mat || frame->mat->empty()) return;

    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        frame_queue_.push(std::move(frame));
    }
    queue_cv_.notify_one();
}

bool VideoRecorder::isRecording() const {
    return recording_.load();
}

std::string VideoRecorder::getCurrentFilePath() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_filepath_;
}

void VideoRecorder::encodingLoop() {
    while (!stop_flag_) {
        std::shared_ptr<core::VideoFramePacket> frame;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !frame_queue_.empty() || stop_flag_; });
            if (stop_flag_ && frame_queue_.empty()) break;
            if (frame_queue_.empty()) continue;
            frame = std::move(frame_queue_.front());
            frame_queue_.pop();
        }

        if (frame && frame->mat && !frame->mat->empty() && encoder_) {
            encoder_->encodeFrame(frame->mat->data, frame->width, frame->height,
                                  static_cast<int>(frame->mat->step), pts_++);
        }
    }
}

bool VideoRecorder::initEncoder(const std::string& filepath, int width, int height) {
    encoder_ = std::make_unique<FileEncoder>();
    return encoder_->init(filepath, "mp4", width, height, bitrate_, "libx264");
}

void VideoRecorder::closeEncoder() {
    if (encoder_) {
        encoder_->close();
        encoder_.reset();
    }
    recording_ = false;
}

} // namespace nodes
} // namespace ai_stream
