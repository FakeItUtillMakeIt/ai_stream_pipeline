// src/nodes/track/bytetrack_adapter.cpp

#include "bytetrack_adapter.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "tracker_factory.h"

namespace ai_stream {
namespace nodes {

ByteTrackAdapter::ByteTrackAdapter(const ByteTrackConfig& config) : config_(config) {
    impl_ = std::make_unique<ai::ByteTrack::BYTETracker>(
        config_.frame_rate, 
        config_.track_buffer
    );
    
    // 设置内部阈值（如果 BYTETracker 有对应的 setter）
    // 如果没有，可能需要修改 BYTETracker 源码或使用默认值
    
    LOG_INFO_FMT("[ByteTrackAdapter] Created: frame_rate={}, track_buffer={}, "
                 "track_thresh={:.2f}, high_thresh={:.2f}, match_thresh={:.2f}",
                 config_.frame_rate, config_.track_buffer,
                 config_.track_thresh, config_.high_thresh, config_.match_thresh);
}

ByteTrackAdapter::~ByteTrackAdapter() {
    LOG_DEBUG("[ByteTrackAdapter] Destroyed");
}

void ByteTrackAdapter::updateConfig(const ByteTrackConfig& config) {
    config_ = config;
    impl_ = std::make_unique<ai::ByteTrack::BYTETracker>(
        config_.frame_rate, 
        config_.track_buffer
    );
    LOG_INFO("[ByteTrackAdapter] Config updated");
}

std::vector<UnifiedTrackResult> ByteTrackAdapter::update(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {
    
    std::vector<UnifiedTrackResult> results;
    
    // 转换检测框格式
    ai::cvUtil::PoseBoxArray objects = convertDetections(detections);
    
    // 调用 ByteTrack 更新
    std::vector<ai::ByteTrack::STrack> tracks = impl_->update(objects);
    
    // 转换跟踪结果
    for (const auto& track : tracks) {
        if (track.is_activated) {
            results.push_back(convertTrack(track));
        }
    }
    
    return results;
}

int ByteTrackAdapter::getActiveCount() const {
    if (!impl_) return 0;
    return impl_->tracked_stracks.size();
}

void ByteTrackAdapter::reset() {
    // 重新创建实例来重置
    impl_ = std::make_unique<ai::ByteTrack::BYTETracker>(
        config_.frame_rate, 
        config_.track_buffer
    );
    LOG_INFO("[ByteTrackAdapter] Reset");
}

ai::cvUtil::PoseBoxArray ByteTrackAdapter::convertDetections(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {
    
    ai::cvUtil::PoseBoxArray objects;
    
    for (const auto& det : detections) {
        if (det.confidence < config_.track_thresh) continue;
        
        ai::cvUtil::PoseBox box;
        box.x = det.x;
        box.y = det.y;
        box.w = det.w;
        box.h = det.h;
        box.confidence = det.confidence;
        box.class_id = det.class_id;
        objects.push_back(box);
    }
    
    return objects;
}

UnifiedTrackResult ByteTrackAdapter::convertTrack(const ai::ByteTrack::STrack& track) {
    UnifiedTrackResult tr;
    tr.x = track.tlwh[0];
    tr.y = track.tlwh[1];
    tr.w = track.tlwh[2];
    tr.h = track.tlwh[3];
    tr.track_id = track.track_id;
    tr.confidence = track.score;
    tr.age = track.age;
    tr.active = track.is_activated;
    
    // 从 mean 中提取平滑后的坐标
    tr.smooth_x = track.mean[0];
    tr.smooth_y = track.mean[1];
    tr.smooth_w = track.mean[2];
    tr.smooth_h = track.mean[3];
    
    return tr;
}

REGISTER_TRACKER(BYTETRACK, ByteTrackAdapter)

} // namespace nodes
} // namespace ai_stream