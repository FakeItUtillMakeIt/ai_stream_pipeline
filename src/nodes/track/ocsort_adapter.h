// src/nodes/track/ocsort_adapter.h
#pragma once

#include "base_tracker.h"
#include "OCSort/OCsort.h"
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
    
    // 更新配置
    void updateConfig(const OCSortConfig& config);

private:
    std::unique_ptr<ocsort::OCSort> impl_;
    OCSortConfig config_;
    
    Eigen::MatrixXf convertDetections(
        const std::vector<core::InferenceResultPacket::BBox>& detections);
    UnifiedTrackResult convertTrack(const Eigen::RowVectorXf& track);
    int findTrackAge(int track_id) const;  // 从 trackers 中查找 age
};

} // namespace nodes
} // namespace ai_stream