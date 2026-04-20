// src/nodes/track/bytetrack_adapter.cpp
#include "bytetrack_adapter.h"
#include "tracker_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

// 隔离第三方库
#define fp_t bytetrack_fp_t
#define FP_1 bytetrack_FP_1
#define FP_2 bytetrack_FP_2
#define FP_DYNAMIC bytetrack_FP_DYNAMIC
#include "ByteTrack/byte_tracker.hpp"
#include "ByteTrack/strack.hpp"
#undef fp_t
#undef FP_1
#undef FP_2
#undef FP_DYNAMIC

namespace ai_stream {
namespace nodes {

class ByteTrackAdapter::Impl {
public:
    std::unique_ptr<ai::ByteTrack::BYTETracker> tracker;
    ByteTrackConfig config;
    
    Impl(const ByteTrackConfig& cfg) : config(cfg) {
        tracker = std::make_unique<ai::ByteTrack::BYTETracker>(
            config.frame_rate,
            config.track_buffer
        );
    }
};

ByteTrackAdapter::ByteTrackAdapter(const ByteTrackConfig& config)
    : impl_(std::make_unique<Impl>(config)) {
    LOG_INFO_FMT("[ByteTrackAdapter] Created: frame_rate={}, track_buffer={}",
                 config.frame_rate, config.track_buffer);
}

ByteTrackAdapter::~ByteTrackAdapter() = default;

void ByteTrackAdapter::updateConfig(const ByteTrackConfig& config) {
    impl_ = std::make_unique<Impl>(config);
}

std::vector<UnifiedTrackResult> ByteTrackAdapter::update(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {
    
    std::vector<UnifiedTrackResult> results;
    
    ai::cvUtil::PoseBoxArray objects;
    for (const auto& det : detections) {
        if (det.confidence < impl_->config.track_thresh) continue;
        
        ai::cvUtil::PoseBox box;
        box.left = det.x - det.w / 2;
        box.right = det.x + det.w / 2;
        box.top = det.y - det.h / 2;
        box.bottom = det.y + det.h / 2;
        box.confidence = det.confidence;
        box.class_label = det.class_id;
        objects.push_back(box);
    }
    
    auto tracks = impl_->tracker->update(objects);
    active_count_ = static_cast<int>(tracks.size());
    
    for (const auto& track : tracks) {
        if (track.is_activated) {
            UnifiedTrackResult tr;
            tr.x = track.tlwh[0];
            tr.y = track.tlwh[1];
            tr.w = track.tlwh[2];
            tr.h = track.tlwh[3];
            tr.track_id = track.track_id;
            tr.confidence = track.score;
            tr.age = track.tracklet_len;
            tr.active = track.is_activated;
            tr.smooth_x = track.mean[0];
            tr.smooth_y = track.mean[1];
            tr.smooth_w = track.mean[2];
            tr.smooth_h = track.mean[3];
            results.push_back(tr);
        }
    }
    
    return results;
}

int ByteTrackAdapter::getActiveCount() const {
    return active_count_;
}

void ByteTrackAdapter::reset() {
    impl_ = std::make_unique<Impl>(impl_->config);
}

REGISTER_TRACKER(BYTETRACK, ByteTrackAdapter)

} // namespace nodes
} // namespace ai_stream