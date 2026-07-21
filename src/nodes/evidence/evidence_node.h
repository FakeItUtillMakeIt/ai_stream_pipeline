// src/nodes/evidence/evidence_node.h
#pragma once

#include "ai_stream/nodes/i_evidence_node.h"
#include "frame_buffer.h"
#include "video_recorder.h"
#include "video_rollover.h"
#include "ftp_uploader.h"
#include "ai_stream/core/packet.h"
#include "ai_stream/rules/i_alert_rule.h"
#include <atomic>
#include <mutex>
#include <string>

namespace ai_stream {
namespace nodes {

class EvidenceNode : public IEvidenceNode {
public:
    EvidenceNode();
    ~EvidenceNode() override;

    bool start() override;
    void stop() override;
    bool isRunning() const override { return running_.load(); }
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    void setVideoConfig(const EvidenceVideoConfig& config) override;
    void setSnapshotConfig(const EvidenceSnapshotConfig& config) override;
    void setRolloverConfig(const EvidenceRolloverConfig& config) override;
    void setFtpConfig(const EvidenceFtpConfig& config) override;

private:
    void handleFrame(std::shared_ptr<core::VideoFramePacket> frame);
    void handleAlertTrigger(std::shared_ptr<core::InferenceResultPacket> packet);

    void startRecording(const rules::AlertEvent& event);
    void stopRecording();

    void saveSnapshotImage(std::shared_ptr<core::VideoFramePacket> frame,
                           const rules::AlertEvent& event);

    std::string generateFilename(const std::string& alert_name, const std::string& ext) const;
    std::string generateTimestamp() const;

    EvidenceVideoConfig video_config_;
    EvidenceSnapshotConfig snapshot_config_;
    EvidenceRolloverConfig rollover_config_;
    EvidenceFtpConfig ftp_config_;

    FrameBuffer frame_buffer_;
    VideoRecorder video_recorder_;
    VideoRollover video_rollover_;
#ifdef WITH_FTP
    FtpConfig ftp_internal_config_;
    std::unique_ptr<FtpUploader> ftp_uploader_;
#endif

    std::atomic<bool> running_{false};
    std::atomic<bool> recording_{false};
    std::atomic<size_t> post_frame_count_{0};
    size_t post_frames_target_ = 40;

    mutable std::mutex mutex_;
};

} // namespace nodes
} // namespace ai_stream
