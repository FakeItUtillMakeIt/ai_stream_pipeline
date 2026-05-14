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

void TrackerNode::setSubStreamId(const std::string& stream_id) {
    sub_stream_id_ = stream_id;
}

void TrackerNode::setTrackerId(const std::string& id) {
    tracker_id_ = id;
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
    in_time_ms_ = utils::TimeUtil::currentTimeMs();
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
    // 过滤不是当前追踪器绑定流的包
    if (!sub_stream_id_.empty() && infer_result->source_id != sub_stream_id_) {
        //LOG_INFO_FMT("[TrackerNode] {} Ignored packet from stream {},expected: {}", tracker_id_, infer_result->source_id, sub_stream_id_);
        //broadcast(packet);
        return;
    }
    LOG_INFO_FMT("[TrackerNode] {} Processing packet from stream {},expected: {}", tracker_id_, infer_result->source_id, sub_stream_id_);
        
    // 执行跟踪
    auto tracks = tracker_->update(infer_result->detections);
    LOG_DEBUG_FMT("[TrackerNode] Tracks: {}", tracks.size());
    // 更新检测框的跟踪信息
    matchAndUpdateDetections(infer_result->detections, tracks);
    // 打印跟踪结果
    for (const auto& track : tracks) {
        LOG_DEBUG_FMT("[TrackerNode] Track ID: {}, Class ID: {}, Age: {}, Active: {}",
                     track.track_id, track.class_id, track.age, track.active);
    }
    //打印检测匹配后的结果
    for (const auto& det : infer_result->detections) {
        LOG_DEBUG_FMT("[TrackerNode] Detection: Class ID: {}, Confidence: {}, Track ID: {}, Age: {}, Active: {}",
                     det.class_id, det.confidence, det.track_id, det.track_age, det.track_active);
    }   
    packet->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
    packet->cost_time_map.insert({name_, packet->cost_ms});
    std::string cost_time_str;
    for (const auto& each_node:packet->cost_time_map)
    {
        cost_time_str += each_node.first + ":" + std::to_string(each_node.second) + "ms,";
    }
    LOG_INFO_FMT("{}",cost_time_str);
    //更新数据包
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