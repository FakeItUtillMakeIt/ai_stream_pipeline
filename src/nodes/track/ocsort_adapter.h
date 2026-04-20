// src/nodes/track/ocsort_adapter.h
#pragma once

#include "base_tracker.h"
#include "ai_stream/nodes/i_tracker_node.h"
#include <memory>

namespace ai_stream {
namespace nodes {

class OCSortAdapter : public BaseTracker {
public:
    explicit OCSortAdapter(const OCSortConfig& config = OCSortConfig{});
    ~OCSortAdapter() override;
    
    TrackerType getType() const override { return TrackerType::OCSORT; }
    
    std::vector<UnifiedTrackResult> update(
        const std::vector<core::InferenceResultPacket::BBox>& detections) override;
    
    int getActiveCount() const override;
    void reset() override;
    
    void updateConfig(const OCSortConfig& config);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nodes
} // namespace ai_stream