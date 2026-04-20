// src/nodes/track/bytetrack_adapter.h
#pragma once

#include "base_tracker.h"
#include "ai_stream/nodes/i_tracker_node.h"
#include <memory>

namespace ai_stream {
namespace nodes {

class ByteTrackAdapter : public BaseTracker {
public:
    explicit ByteTrackAdapter(const ByteTrackConfig& config = ByteTrackConfig{});
    ~ByteTrackAdapter() override;
    
    TrackerType getType() const override { return TrackerType::BYTETRACK; }
    
    std::vector<UnifiedTrackResult> update(
        const std::vector<core::InferenceResultPacket::BBox>& detections) override;
    
    int getActiveCount() const override;
    void reset() override;
    
    void updateConfig(const ByteTrackConfig& config);

private:
    int active_count_ = 0;
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nodes
} // namespace ai_stream