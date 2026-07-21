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
};

} // namespace nodes
} // namespace ai_stream
