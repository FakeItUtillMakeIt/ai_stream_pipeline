// src/nodes/track/ocsort_adapter.cpp

#include "ocsort_adapter.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "tracker_factory.h"
#include <Eigen/Dense>

namespace ai_stream {
namespace nodes {

OCSortAdapter::OCSortAdapter(const OCSortConfig& config) : config_(config) {
    impl_ = std::make_unique<ocsort::OCSort>(
        config_.det_thresh,
        config_.max_age,
        config_.min_hits,
        config_.iou_threshold,
        config_.delta_t,
        config_.asso_func,
        config_.inertia,
        config_.use_byte
    );
    
    LOG_INFO_FMT("[OCSortAdapter] Created: det_thresh={:.2f}, max_age={}, min_hits={}, "
                 "iou_thresh={:.2f}, delta_t={}, asso_func={}, inertia={:.2f}, use_byte={}",
                 config_.det_thresh, config_.max_age, config_.min_hits,
                 config_.iou_threshold, config_.delta_t, config_.asso_func,
                 config_.inertia, config_.use_byte);
}

OCSortAdapter::~OCSortAdapter() {
    LOG_DEBUG("[OCSortAdapter] Destroyed");
}

void OCSortAdapter::updateConfig(const OCSortConfig& config) {
    config_ = config;
    impl_ = std::make_unique<ocsort::OCSort>(
        config_.det_thresh, config_.max_age, config_.min_hits,
        config_.iou_threshold, config_.delta_t, config_.asso_func,
        config_.inertia, config_.use_byte
    );
    LOG_INFO("[OCSortAdapter] Config updated");
}

std::vector<UnifiedTrackResult> OCSortAdapter::update(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {
    
    std::vector<UnifiedTrackResult> results;
    
    // 过滤低置信度检测
    std::vector<core::InferenceResultPacket::BBox> filtered_dets;
    for (const auto& det : detections) {
        if (det.confidence >= config_.det_thresh) {
            filtered_dets.push_back(det);
        }
    }
    
    if (filtered_dets.empty()) {
        // 即使没有检测，也调用 update 让轨迹衰减
        Eigen::MatrixXf empty_dets(0, 6);
        impl_->update(empty_dets);
        return results;
    }
    
    // 转换检测框格式
    Eigen::MatrixXf dets = convertDetections(filtered_dets);
    
    // 调用 OCSort 更新
    std::vector<Eigen::RowVectorXf> tracks = impl_->update(dets);
    
    // 转换跟踪结果
    for (const auto& track : tracks) {
        // 检查是否为有效结果（非全零）
        if (track.size() >= 7 && track(4) > 0) {
            results.push_back(convertTrack(track));
        }
    }
    
    return results;
}

int OCSortAdapter::getActiveCount() const {
    if (!impl_) return 0;
    
    int active_count = 0;
    for (const auto& tracker : impl_->trackers) {
        if (tracker.hits >= config_.min_hits) {
            active_count++;
        }
    }
    return active_count;
}

void OCSortAdapter::reset() {
    if (impl_) {
        impl_->trackers.clear();
        impl_->frame_count = 0;
    }
    LOG_INFO("[OCSortAdapter] Reset");
}

Eigen::MatrixXf OCSortAdapter::convertDetections(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {
    
    // 格式：[x1, y1, x2, y2, confidence, class_id]
    Eigen::MatrixXf dets(detections.size(), 6);
    
    for (size_t i = 0; i < detections.size(); ++i) {
        const auto& det = detections[i];
        dets(i, 0) = det.x;
        dets(i, 1) = det.y;
        dets(i, 2) = det.x + det.w;
        dets(i, 3) = det.y + det.h;
        dets(i, 4) = det.confidence;
        dets(i, 5) = static_cast<float>(det.class_id);
    }
    
    return dets;
}

UnifiedTrackResult OCSortAdapter::convertTrack(const Eigen::RowVectorXf& track) {
    // 返回格式：[x1, y1, x2, y2, track_id, class_id, confidence]
    UnifiedTrackResult tr;
    tr.x = track(0);
    tr.y = track(1);
    tr.w = track(2) - track(0);
    tr.h = track(3) - track(1);
    tr.track_id = static_cast<int>(track(4));
    tr.class_id = static_cast<int>(track(5));
    tr.confidence = track(6);
    tr.age = findTrackAge(tr.track_id);
    tr.active = (tr.age < config_.max_age);
    
    return tr;
}

int OCSortAdapter::findTrackAge(int track_id) const {
    for (const auto& tracker : impl_->trackers) {
        if (tracker.id == track_id) {
            return tracker.age;
        }
    }
    return 0;
}

REGISTER_TRACKER(OCSORT, OCSortAdapter)

} // namespace nodes
} // namespace ai_stream