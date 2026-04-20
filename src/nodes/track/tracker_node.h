// src/nodes/track/tracker_node.h
#pragma once

#include "ai_stream/nodes/i_tracker_node.h"
#include "base_tracker.h"
#include <atomic>
#include <memory>

namespace ai_stream {
namespace nodes {

class TrackerNode : public ITrackerNode {
public:
    TrackerNode();
    ~TrackerNode() override;
    
    // ITrackerNode 接口
    void setTrackerType(TrackerType type) override;
    TrackerType getTrackerType() const override;
    void setOCSortConfig(const OCSortConfig& config) override;
    void setByteTrackConfig(const ByteTrackConfig& config) override;
    int getActiveTrackCount() const override;
    
    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void matchAndUpdateDetections(std::vector<core::InferenceResultPacket::BBox>& detections,
                                   const std::vector<UnifiedTrackResult>& tracks);
    
    TrackerType tracker_type_ = TrackerType::OCSORT;
    OCSortConfig ocsort_config_;
    ByteTrackConfig bytetrack_config_;
    TrackerPtr tracker_;
    std::atomic<bool> running_{false};
};

} // namespace nodes
} // namespace ai_stream