// src/nodes/evidence/video_recorder.h
#pragma once

#include "ai_stream/core/packet.h"
#include "ai_stream/core/bounded_queue.h"
#include "sink/encoder_base.h"
#include <memory>
#include <string>
#include <deque>
#include <atomic>
#include <condition_variable>
#include <thread>
#include <mutex>

namespace ai_stream {
namespace nodes {

/**
 * @brief 证据录像器
 *
 * 常驻一个编码线程；编码器（含 HAL/NVENC 上下文）在多次告警录制之间复用，
 * 每次录制只切换 MP4 封装输出，避免频繁初始化/销毁硬件编码器造成的
 * 内存与线程抖动。
 */
class VideoRecorder {
public:
    VideoRecorder();
    ~VideoRecorder();

    bool initialize(const std::string& output_dir, int fps, int bitrate, const std::string& codec);
    bool startRecording(const std::string& filename,
                       std::deque<std::shared_ptr<core::VideoFramePacket>> pre_frames);
    void enqueueFrame(std::shared_ptr<core::VideoFramePacket> frame);
    // 结束当前录制；编码线程与编码器保持常驻
    void stop();
    // 停止编码线程并释放编码器（pipeline 停止/对象销毁时调用）
    void shutdown();
    bool isRecording() const;

    // 日志上下文（由 EvidenceNode 注入，用于编码子线程）
    void setLogContext(const std::string& pipeline, const std::string& node) {
        log_pipeline_ = pipeline;
        log_node_ = node;
    }
    std::string getCurrentFilePath() const;

private:
    void encodingLoop();
    bool startEncoder(const std::string& filepath, int width, int height);
    void finishEncoder();
    void closeEncoder();

    std::string output_dir_;
    int fps_ = 25;
    int bitrate_ = 4000;
    std::string codec_ = "libx264";

    std::atomic<bool> recording_{false};
    std::string current_filepath_;
    std::string log_pipeline_;
    std::string log_node_;
    mutable std::mutex mutex_;

    std::unique_ptr<FileEncoder> encoder_;
    std::atomic<int64_t> pts_{0};
    int width_ = 0;      // 编码器初始化尺寸（以首帧 mat 为准）
    int height_ = 0;

    // 有界队列（满时丢最旧帧，防止编码慢时无界增长）
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> frame_queue_{256};

    // 常驻编码线程及命令握手
    std::thread encode_thread_;
    std::condition_variable cv_;
    bool shutdown_ = false;
    bool start_pending_ = false;
    bool start_ack_ = false;
    bool start_ok_ = false;
    bool stop_requested_ = false;
    bool worker_done_ = true;
    std::string pending_filepath_;
    int pending_width_ = 0;
    int pending_height_ = 0;
};

} // namespace nodes
} // namespace ai_stream
