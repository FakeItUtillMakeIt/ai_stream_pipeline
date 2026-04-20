// src/nodes/track/ocsort_adapter.cpp
#include "ocsort_adapter.h"
#include "tracker_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

// 隔离第三方库，解决 fp_t 冲突
#define fp_t ocsort_fp_t
#define FP_1 ocsort_FP_1
#define FP_2 ocsort_FP_2
#define FP_DYNAMIC ocsort_FP_DYNAMIC
#include "OCSort/OCsort.h"
#include "OCSort/KalmanBoxTracker.h"
#include "OCSort/association.h"
#undef fp_t
#undef FP_1
#undef FP_2
#undef FP_DYNAMIC

#include <Eigen/Dense>

namespace ai_stream {
namespace nodes {

// Pimpl 实现类
class OCSortAdapter::Impl {
public:
    std::unique_ptr<ocsort::OCSort> tracker;
    OCSortConfig config;
    
    Impl(const OCSortConfig& cfg) : config(cfg) {
        tracker = std::make_unique<ocsort::OCSort>(
            config.det_thresh,
            config.max_age,
            config.min_hits,
            config.iou_threshold,
            config.delta_t,
            config.asso_func,
            config.inertia,
            config.use_byte
        );
    }
};

OCSortAdapter::OCSortAdapter(const OCSortConfig& config) 
    : impl_(std::make_unique<Impl>(config)) {
    LOG_INFO_FMT("[OCSortAdapter] Created: det_thresh={:.2f}, max_age={}, min_hits={}",
                 config.det_thresh, config.max_age, config.min_hits);
}

OCSortAdapter::~OCSortAdapter() = default;

void OCSortAdapter::updateConfig(const OCSortConfig& config) {
    impl_ = std::make_unique<Impl>(config);
    LOG_INFO("[OCSortAdapter] Config updated");
}

std::vector<UnifiedTrackResult> OCSortAdapter::update(
    const std::vector<core::InferenceResultPacket::BBox>& detections) {
    
    std::vector<UnifiedTrackResult> results;
    
    // 过滤低置信度检测
    std::vector<core::InferenceResultPacket::BBox> filtered_dets;
    for (const auto& det : detections) {
        if (det.confidence >= impl_->config.det_thresh) {
            filtered_dets.push_back(det);
        }
    }
    
    if (filtered_dets.empty()) {
        Eigen::MatrixXf empty_dets(0, 6);
        impl_->tracker->update(empty_dets);
        return results;
    }
    
    // 转换检测框格式 [x1, y1, x2, y2, confidence, class_id]
    Eigen::MatrixXf dets(filtered_dets.size(), 6);
    for (size_t i = 0; i < filtered_dets.size(); ++i) {
        const auto& det = filtered_dets[i];
        dets(i, 0) = det.x;
        dets(i, 1) = det.y;
        dets(i, 2) = det.x + det.w;
        dets(i, 3) = det.y + det.h;
        dets(i, 4) = det.confidence;
        dets(i, 5) = static_cast<float>(det.class_id);
    }
    
    // 调用 OCSort
    auto tracks = impl_->tracker->update(dets);
    
    // 转换结果
    for (const auto& track : tracks) {
        if (track.size() >= 7 && track(4) > 0) {
            UnifiedTrackResult tr;
            tr.x = track(0);
            tr.y = track(1);
            tr.w = track(2) - track(0);
            tr.h = track(3) - track(1);
            tr.track_id = static_cast<int>(track(4));
            tr.class_id = static_cast<int>(track(5));
            tr.confidence = track(6);
            tr.age = 0;
            tr.active = true;
            results.push_back(tr);
        }
    }
    
    return results;
}

int OCSortAdapter::getActiveCount() const {
    int count = 0;
    for (const auto& t : impl_->tracker->trackers) {
        if (t.hits >= impl_->config.min_hits) count++;
    }
    return count;
}

void OCSortAdapter::reset() {
    impl_->tracker->trackers.clear();
    impl_->tracker->frame_count = 0;
    LOG_INFO("[OCSortAdapter] Reset");
}

// 注册到工厂
REGISTER_TRACKER(OCSORT, OCSortAdapter)

} // namespace nodes
} // namespace ai_stream