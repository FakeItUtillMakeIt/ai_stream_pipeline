// src/nodes/track/tracker_node.cpp
#include "tracker_node.h"
#include "tracker_factory.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

// 注意：不包含 ocsort_adapter.h 和 bytetrack_adapter.h

namespace ai_stream {
namespace nodes {

TrackerNode::TrackerNode() : ITrackerNode("TrackerNode") {
    LOG_DEBUG("[TrackerNode] Constructor");
}

TrackerNode::~TrackerNode() {
    stop();
    LOG_DEBUG("[TrackerNode] Destructor");
}

void TrackerNode::setTrackerType(TrackerType type) {
    tracker_type_ = type;
    LOG_INFO_FMT("[TrackerNode] Tracker type: {}", 
                 type == TrackerType::OCSORT ? "OCSORT" : "BYTETRACK");
}

TrackerType TrackerNode::getTrackerType() const {
    return tracker_type_;
}

void TrackerNode::setOCSortConfig(const OCSortConfig& config) {
    ocsort_config_ = config;
}

void TrackerNode::setByteTrackConfig(const ByteTrackConfig& config) {
    bytetrack_config_ = config;
}

int TrackerNode::getActiveTrackCount() const {
    return tracker_ ? tracker_->getActiveCount() : 0;
}

bool TrackerNode::start() {
    // 通过工厂创建跟踪器
    tracker_ = TrackerFactory::instance().create(tracker_type_);
    if (!tracker_) {
        LOG_ERROR_FMT("[TrackerNode] Failed to create tracker");
        return false;
    }
    
    running_ = true;
    LOG_INFO_FMT("[TrackerNode] Started with {} tracker", 
                 tracker_type_ == TrackerType::OCSORT ? "OCSORT" : "BYTETRACK");
    return true;
}

void TrackerNode::stop() {
    running_ = false;
    if (tracker_) {
        tracker_->reset();
    }
    LOG_INFO_FMT("[TrackerNode] Stopped");
}

void TrackerNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (!running_) return;
    
    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }
    
    auto infer_result = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer_result) {
        broadcast(packet);
        return;
    }
    
    // 执行跟踪
    auto tracks = tracker_->update(infer_result->detections);
    
    // 更新检测框的跟踪信息
    matchAndUpdateDetections(infer_result->detections, tracks);
    
    broadcast(packet);
}

void TrackerNode::matchAndUpdateDetections(
    std::vector<core::InferenceResultPacket::BBox>& detections,
    const std::vector<UnifiedTrackResult>& tracks) {
    
    for (auto& det : detections) {
        det.track_id = -1;
        det.track_age = 0;
        det.track_active = false;
        
        for (const auto& track : tracks) {
            float ix1 = std::max(det.x, track.x);
            float iy1 = std::max(det.y, track.y);
            float ix2 = std::min(det.x + det.w, track.x + track.w);
            float iy2 = std::min(det.y + det.h, track.y + track.h);
            
            if (ix2 <= ix1 || iy2 <= iy1) continue;
            
            float intersection = (ix2 - ix1) * (iy2 - iy1);
            float area_det = det.w * det.h;
            float area_track = track.w * track.h;
            float iou = intersection / (area_det + area_track - intersection);
            
            if (iou > 0.5f && track.class_id == det.class_id) {
                det.track_id = track.track_id;
                det.track_age = track.age;
                det.track_active = track.active;
                det.smooth_x = track.smooth_x;
                det.smooth_y = track.smooth_y;
                det.smooth_w = track.smooth_w;
                det.smooth_h = track.smooth_h;
                break;
            }
        }
    }
}

// 注册节点
REGISTER_NODE("tracker", TrackerNode)

} // namespace nodes
} // namespace ai_stream