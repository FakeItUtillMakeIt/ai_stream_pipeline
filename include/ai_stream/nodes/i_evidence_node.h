// include/ai_stream/nodes/i_evidence_node.h
#pragma once

#include "ai_stream/core/node.h"
#include <string>

namespace ai_stream {
namespace nodes {

struct EvidenceVideoConfig {
    bool enabled = true;
    int pre_frames = 20;
    int post_frames = 40;
    int fps = 25;
    int bitrate = 4000;
    std::string output_dir = "./evidence/videos";
};

struct EvidenceSnapshotConfig {
    bool enabled = true;
    std::string output_dir = "./evidence/snapshots";
};

struct EvidenceRolloverConfig {
    bool enabled = true;
    uint32_t retention_hours = 72;
    uint32_t check_interval_min = 60;
};

struct EvidenceFtpConfig {
    bool enabled = false;
    std::string server;
    int port = 21;
    std::string username;
    std::string password;
    std::string remote_dir = "/evidence";
    bool use_tls = false;
    bool delete_after_upload = false;
    int max_retry = 3;
};

class IEvidenceNode : public core::Node {
public:
    using core::Node::Node;

    virtual void setVideoConfig(const EvidenceVideoConfig& config) = 0;
    virtual void setSnapshotConfig(const EvidenceSnapshotConfig& config) = 0;
    virtual void setRolloverConfig(const EvidenceRolloverConfig& config) = 0;
    virtual void setFtpConfig(const EvidenceFtpConfig& config) = 0;

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        (void)node_id;
        const nlohmann::json video_cfg = params.value("video", nlohmann::json::object());
        setVideoConfig(EvidenceVideoConfig{
            .enabled = video_cfg.value("enabled", true),
            .pre_frames = video_cfg.value("pre_frames", 10),
            .post_frames = video_cfg.value("post_frames", 10),
            .fps = video_cfg.value("fps", 30),
            .bitrate = video_cfg.value("bitrate", 1000),
            .output_dir = video_cfg.value("output_dir", "./")
        });

        const nlohmann::json snapshot_cfg = params.value("snapshot", nlohmann::json::object());
        setSnapshotConfig(EvidenceSnapshotConfig{
            .enabled = snapshot_cfg.value("enabled", false),
            .output_dir = snapshot_cfg.value("output_dir", "./")
        });

        const nlohmann::json rollover_cfg = params.value("rollover", nlohmann::json::object());
        setRolloverConfig(EvidenceRolloverConfig{
            .enabled = rollover_cfg.value("enabled", false),
            .retention_hours = static_cast<uint32_t>(rollover_cfg.value("retention_hours", 24)),
            .check_interval_min = static_cast<uint32_t>(rollover_cfg.value("check_interval_min", 60))
        });

        const nlohmann::json ftp_cfg = params.value("ftp", nlohmann::json::object());
        EvidenceFtpConfig ftp_config{
            .enabled = ftp_cfg.value("enabled", false),
            .server = ftp_cfg.value("server", ""),
            .port = ftp_cfg.value("port", 21),
            .username = ftp_cfg.value("username", ""),
            .password = ftp_cfg.value("password", ""),
            .remote_dir = ftp_cfg.value("remote_dir", ""),
        };
        ftp_config.use_tls = ftp_cfg.value("use_tls", false);
        ftp_config.delete_after_upload = ftp_cfg.value("delete_after_upload", false);
        ftp_config.max_retry = ftp_cfg.value("max_retry", 3);
        setFtpConfig(ftp_config);
        return true;
    }
};

} // namespace nodes
} // namespace ai_stream
