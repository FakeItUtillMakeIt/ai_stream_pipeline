// src/nodes/evidence/evidence_node.cpp
#include "evidence_node.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#ifdef WITH_FTP
#include "ftp_uploader.h"
#endif
#include <opencv2/opencv.hpp>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <filesystem>

namespace ai_stream {
namespace nodes {

EvidenceNode::EvidenceNode() : IEvidenceNode("EvidenceNode") {
    LOG_INFO("[EvidenceNode] Constructor");
}

EvidenceNode::~EvidenceNode() {
    stop();
    LOG_DEBUG("[EvidenceNode] Destructor");
}

bool EvidenceNode::start() {
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = true;

    if (video_config_.enabled) {
        if (!video_recorder_.initialize(video_config_.output_dir, video_config_.fps, video_config_.bitrate)) {
            LOG_ERROR("[EvidenceNode] Failed to initialize video recorder");
            return false;
        }
        frame_buffer_.setCapacity(video_config_.pre_frames);
        post_frames_target_ = video_config_.post_frames;
    }

    if (snapshot_config_.enabled) {
        std::error_code ec;
        std::filesystem::create_directories(snapshot_config_.output_dir, ec);
        if (ec) {
            LOG_ERROR_FMT("[EvidenceNode] Failed to create snapshot directory: {}", ec.message());
            return false;
        }
    }

    if (rollover_config_.enabled) {
        VideoRolloverConfig rcfg;
        rcfg.video_dir = video_config_.output_dir;
        rcfg.retention_hours = rollover_config_.retention_hours;
        rcfg.check_interval_min = rollover_config_.check_interval_min;
        rcfg.enabled = true;
        video_rollover_.configure(rcfg);
        video_rollover_.start();
    }

    if (ftp_config_.enabled) {
#ifdef WITH_FTP
        ftp_internal_config_.server = ftp_config_.server;
        ftp_internal_config_.port = ftp_config_.port;
        ftp_internal_config_.username = ftp_config_.username;
        ftp_internal_config_.password = ftp_config_.password;
        ftp_internal_config_.remote_dir = ftp_config_.remote_dir;
        ftp_internal_config_.use_tls = ftp_config_.use_tls;
        ftp_internal_config_.delete_after_upload = ftp_config_.delete_after_upload;
        ftp_internal_config_.max_retry = ftp_config_.max_retry;
        ftp_uploader_ = std::make_unique<FtpUploader>();
        if (!ftp_uploader_->initialize(ftp_internal_config_)) {
            LOG_ERROR("[EvidenceNode] Failed to initialize FTP uploader");
            ftp_uploader_.reset();
        }
#else
        LOG_WARN("[EvidenceNode] FTP feature disabled at build time, skipping FTP initialization");
#endif
    }

    LOG_INFO_FMT("[EvidenceNode] Started (video={}, snapshot={}, rollover={}, ftp={})",
                 video_config_.enabled, snapshot_config_.enabled,
                 rollover_config_.enabled, ftp_config_.enabled);
    return true;
}

void EvidenceNode::stop() {
    if (!running_.exchange(false)) {
        return;
    }

    {
        std::lock_guard<std::mutex> lock(pending_snapshot_mutex_);
        pending_snapshot_event_.reset();
    }

    video_recorder_.stop();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        recording_ = false;
    }

    video_rollover_.stop();

#ifdef WITH_FTP
    if (ftp_uploader_) {
        ftp_uploader_->shutdown();
        ftp_uploader_.reset();
    }
#endif

    LOG_INFO("[EvidenceNode] Stopped");
}

void EvidenceNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO("[EvidenceNode] Received stream end");
        stop();
        return;
    }

    if (!running_) return;

    if (packet->type == core::PacketType::DECODED_FRAME) {
        // 来自 draw node 的绘制后帧
        auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
        handleFrame(std::move(frame));
    } else if (packet->type == core::PacketType::META_DATA) {
        // 来自 alert node 的告警触发
        auto infer = std::static_pointer_cast<core::InferenceResultPacket>(packet);
        if (infer) {
            handleAlertTrigger(std::move(infer));
        }
    }
}

void EvidenceNode::handleFrame(std::shared_ptr<core::VideoFramePacket> frame) {
    if (!frame || !frame->mat || frame->mat->empty()) return;

    if (video_config_.enabled) {
        frame_buffer_.push(frame);
    }

    trySavePendingSnapshot(frame);

    if (recording_) {
        video_recorder_.enqueueFrame(frame);
        post_frame_count_++;

        if (post_frame_count_ >= post_frames_target_) {
            stopRecording();
        }
    }
}

void EvidenceNode::handleAlertTrigger(std::shared_ptr<core::InferenceResultPacket> packet) {
    if (!packet) return;

    for (const auto& result : packet->alert_result) {
        for (const auto& event : result.alert_events) {
            if (event.status == rules::AlertStatus::ALERT_STATUS_OCCUR) {
                LOG_INFO_FMT("[EvidenceNode] Alert triggered: {} ({})",
                             event.alert_name, rules::alertTypeMap[event.alert_type]);

                if (snapshot_config_.enabled) {
                    std::lock_guard<std::mutex> lock(pending_snapshot_mutex_);
                    pending_snapshot_event_ = event;
                }

                if (video_config_.enabled) {
                    startRecording(event);
                }
            }
        }
    }
}

void EvidenceNode::trySavePendingSnapshot(std::shared_ptr<core::VideoFramePacket> frame) {
    if (!snapshot_config_.enabled || !frame || !frame->mat || frame->mat->empty()) return;

    std::optional<rules::AlertEvent> event;
    {
        std::lock_guard<std::mutex> lock(pending_snapshot_mutex_);
        if (pending_snapshot_event_) {
            event = pending_snapshot_event_;
            pending_snapshot_event_.reset();
        }
    }

    if (event) {
        saveSnapshotImage(frame, *event);
    }
}

void EvidenceNode::startRecording(const rules::AlertEvent& event) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (recording_) {
        LOG_WARN("[EvidenceNode] Already recording, ignoring new trigger");
        return;
    }

    if (!video_config_.enabled) return;

    std::string filename = generateFilename(event.alert_name, "mp4");
    auto pre_frames = frame_buffer_.snapshot();
    size_t pre_frames_count = pre_frames.size();

    if (pre_frames.empty()) {
        LOG_WARN("[EvidenceNode] No pre-frames available, skipping recording");
        return;
    }

    if (video_recorder_.startRecording(filename, std::move(pre_frames))) {
        recording_ = true;
        post_frame_count_ = 0;
        LOG_INFO_FMT("[EvidenceNode] Started recording: {} ({} pre-frames)", filename, pre_frames_count);
    }
}

void EvidenceNode::stopRecording() {
    std::string filepath;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!recording_) return;
        filepath = video_recorder_.getCurrentFilePath();
        recording_ = false;
    }

    video_recorder_.stop();

    LOG_INFO_FMT("[EvidenceNode] Recording complete: {}", filepath);

#ifdef WITH_FTP
    if (ftp_uploader_ && !filepath.empty()) {
        ftp_uploader_->enqueue(filepath);
    }
#endif
}

void EvidenceNode::saveSnapshotImage(std::shared_ptr<core::VideoFramePacket> frame,
                                     const rules::AlertEvent& event) {
    if (!frame || !frame->mat || frame->mat->empty()) return;

    try {
        std::string filename = generateFilename(event.alert_name, "jpg");
        std::filesystem::path filepath = snapshot_config_.output_dir;
        filepath /= filename;

        cv::imwrite(filepath.string(), *frame->mat);
        LOG_INFO_FMT("[EvidenceNode] Snapshot saved: {}", filepath.string());
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[EvidenceNode] Failed to save snapshot: {}", e.what());
    }
}

std::string EvidenceNode::generateFilename(const std::string& alert_name, const std::string& ext) const {
    return alert_name + "_" + generateTimestamp() + "." + ext;
}

std::string EvidenceNode::generateTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;

    std::tm tm;
    localtime_r(&tt, &tm);

    std::ostringstream ss;
    ss << std::put_time(&tm, "%Y%m%d_%H%M%S")
       << "_" << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

void EvidenceNode::setVideoConfig(const EvidenceVideoConfig& config) {
    video_config_ = config;
}

void EvidenceNode::setSnapshotConfig(const EvidenceSnapshotConfig& config) {
    snapshot_config_ = config;
}

void EvidenceNode::setRolloverConfig(const EvidenceRolloverConfig& config) {
    rollover_config_ = config;
}

void EvidenceNode::setFtpConfig(const EvidenceFtpConfig& config) {
    ftp_config_ = config;
}

REGISTER_NODE("evidence", EvidenceNode)

} // namespace nodes
} // namespace ai_stream
