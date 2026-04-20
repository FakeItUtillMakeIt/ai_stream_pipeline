// src/nodes/track/bytetrack_adapter.h
#pragma once

#include "base_tracker.h"
#include "ByteTrack/byte_tracker.hpp"
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
    std::unique_ptr<ai::ByteTrack::BYTETracker> impl_;
    ByteTrackConfig config_;
    
    ai::cvUtil::PoseBoxArray convertDetections(
        const std::vector<core::InferenceResultPacket::BBox>& detections);
    UnifiedTrackResult convertTrack(const ai::ByteTrack::STrack& track);
};

} // namespace nodes
} // namespace ai_stream