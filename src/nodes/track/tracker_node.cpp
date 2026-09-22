// src/nodes/track/tracker_node.cpp
#include "tracker_node.h"
#include "tracker_factory.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <algorithm>
#include <cmath>
#include <unordered_set>

// 注意：不包含 ocsort_adapter.h 和 bytetrack_adapter.h

namespace ai_stream {
namespace nodes {

TrackerNode::TrackerNode() : core::QueuedNode<ITrackerNode>("TrackerNode") {
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

bool TrackerNode::configureImpl(const std::string& node_id, const nlohmann::json& params) {
    if (params.contains("stitch") && params["stitch"].is_object()) {
        const auto& s = params["stitch"];
        stitch_enabled_ = s.value("enabled", stitch_enabled_);
        stitch_gap_frames_ = s.value("gap_frames", stitch_gap_frames_);
        stitch_dist_ratio_ = s.value("dist_ratio", stitch_dist_ratio_);
        stitch_memory_frames_ = s.value("memory_frames", stitch_memory_frames_);
        LOG_INFO_FMT("[TrackerNode] Stitch config: enabled={}, gap={}, ratio={}, memory={}",
                     stitch_enabled_, stitch_gap_frames_, stitch_dist_ratio_, stitch_memory_frames_);
    }
    return ITrackerNode::configure(node_id, params);
}

bool TrackerNode::onStartup() {
    // 通过工厂创建跟踪器
    tracker_ = TrackerFactory::instance().create(tracker_type_);
    if (!tracker_) {
        LOG_ERROR_FMT("[TrackerNode] Failed to create tracker");
        return false;
    }

    LOG_INFO_FMT("[TrackerNode] Started with {} tracker",
                 tracker_type_ == TrackerType::OCSORT ? "OCSORT" : "BYTETRACK");
    return true;
}

void TrackerNode::onShutdown() {
    if (tracker_) {
        tracker_->reset();
    }
    track_class_names_.clear();
    id_remap_.clear();
    raw_last_frame_.clear();
    track_memory_.clear();
    stitch_frame_counter_ = 0;
    LOG_INFO_FMT("[TrackerNode] Stopped");
}

void TrackerNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[Tracker] Received stream end");
        // STREAM_END：由 worker 处理完本帧后基类统一 stop()；此处仅转发(broadcast)
        broadcast(packet);
        return;
    }
    in_time_ms_ = utils::TimeUtil::currentTimeMs();

    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }

    auto infer_result = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer_result) {
        broadcast(packet);
        return;
    }
    // 过滤不是当前追踪器绑定流的包（非本流透传，避免多路分支下静默丢包）
    if (!sub_stream_id_.empty() && infer_result->source_id != sub_stream_id_) {
        broadcast(packet);
        return;
    }
    LOG_DEBUG_FMT("[TrackerNode] {} Processing packet from stream {},expected: {}", tracker_id_, infer_result->source_id, sub_stream_id_);
        
    // 执行跟踪
    auto tracks = tracker_->update(infer_result->detections);

    // 轨迹 ID 缝合：保持目标在短暂丢失/类别跃迁后的 ID 稳定
    stitch_frame_counter_++;
    applyStitching(tracks, stitch_frame_counter_);
    cleanupStitchState(stitch_frame_counter_);

    // 收集当前活跃的轨迹 ID
    std::unordered_set<int> current_active_ids;
    for (const auto& track : tracks) {
        current_active_ids.insert(track.track_id);
    }

    // 清理过期的 track_class_names_ 条目（防止内存无限增长）
    if (track_class_names_.size() > MAX_TRACK_CLASS_NAMES) {
        auto it = track_class_names_.begin();
        while (it != track_class_names_.end()) {
            if (current_active_ids.count(it->first) == 0) {
                it = track_class_names_.erase(it);
            } else {
                ++it;
            }
        }
    }

    // 为新轨迹绑定 class_name，同时完成检测框匹配
    for (auto& det : infer_result->detections) {
        det.track_id = -1;
        det.track_age = 0;
        det.track_active = false;
    }

    const int num_dets = static_cast<int>(infer_result->detections.size());
    const int num_tracks = static_cast<int>(tracks.size());

    // 预计算 IoU 矩阵 [det][track]，供类别绑定与匹配复用，避免 O(D×T) 重复计算
    std::vector<std::vector<float>> iou_mat(num_dets, std::vector<float>(num_tracks, 0.0f));
    for (int d = 0; d < num_dets; ++d) {
        for (int t = 0; t < num_tracks; ++t) {
            iou_mat[d][t] = computeIoU(infer_result->detections[d], tracks[t]);
        }
    }

    for (int t = 0; t < num_tracks; ++t) {
        const auto& track = tracks[t];

        // 找与当前轨迹 IoU 最大的检测框（不限类别），用于类别绑定/跃迁判断
        float best_iou = 0.0f;
        int best_d = -1;
        for (int d = 0; d < num_dets; ++d) {
            if (iou_mat[d][t] > best_iou) {
                best_iou = iou_mat[d][t];
                best_d = d;
            }
        }
        const auto* best_det = best_d >= 0 ? &infer_result->detections[best_d] : nullptr;

        auto name_it = track_class_names_.find(track.track_id);
        if (name_it == track_class_names_.end()) {
            // 新轨迹：首次绑定类别
            if (best_det && best_iou > 0.3f && !best_det->class_name.empty()) {
                track_class_names_[track.track_id] = {best_det->class_id, best_det->class_name};
            }
        } else if (best_det && best_iou > 0.5f &&
                   !best_det->class_name.empty() &&
                   name_it->second.name != best_det->class_name &&
                   name_it->second.class_id != best_det->class_id) {
            // 类别跃迁（如 person -> down）：name 与 class_id 同时变化时更新绑定，
            // 保持 track_id 连续。若仅 name 变化（多源 class_id 冲突）则拒绝，交由匹配环节过滤
            LOG_INFO_FMT("[TrackerNode] Track {} class transition: {} -> {}",
                         track.track_id, name_it->second.name, best_det->class_name);
            name_it->second = {best_det->class_id, best_det->class_name};
        }

        // 检测框匹配（使用 track_class_names_ 进行类别匹配）
        for (int d = 0; d < num_dets; ++d) {
            auto& det = infer_result->detections[d];
            if (det.track_id != -1) continue; // 已匹配

            if (iou_mat[d][t] <= 0.5f) continue;

            // 类别匹配
            bool class_match;
            auto it = track_class_names_.find(track.track_id);
            if (it != track_class_names_.end() && !it->second.name.empty() && !det.class_name.empty()) {
                class_match = (it->second.name == det.class_name);
            } else {
                class_match = (track.class_id == det.class_id);
            }

            if (class_match) {
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

    LOG_DEBUG_FMT("[TrackerNode] Tracks: {}", tracks.size());
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
    LOG_DEBUG_FMT("{}",cost_time_str);
    //更新数据包
    broadcast(packet);
}

void TrackerNode::applyStitching(std::vector<UnifiedTrackResult>& tracks, int64_t frame_id) {
    if (!stitch_enabled_) return;

    std::unordered_set<int> taken;
    std::vector<std::pair<size_t, int>> pending;  // (index, raw_id)

    // 第一遍：已登记的原始 id 直接映射，先占位，避免被缝合抢占
    for (size_t i = 0; i < tracks.size(); ++i) {
        const int raw = tracks[i].track_id;
        auto it = id_remap_.find(raw);
        if (it != id_remap_.end()) {
            tracks[i].track_id = it->second;
            taken.insert(it->second);
            raw_last_frame_[raw] = frame_id;
        } else {
            pending.emplace_back(i, raw);
        }
    }

    // 第二遍：未登记的原始 id 尝试缝合到最近丢失且未被占用的轨迹
    for (const auto& p : pending) {
        auto& track = tracks[p.first];
        const int raw = p.second;
        int best = -1;
        float best_dist = 0.0f;
        const float cx1 = track.x + track.w / 2;
        const float cy1 = track.y + track.h / 2;
        for (const auto& kv : track_memory_) {
            const int cid = kv.first;
            if (taken.count(cid)) continue;
            if (frame_id - kv.second.last_frame_id > stitch_gap_frames_) continue;
            const float cx2 = kv.second.x + kv.second.w / 2;
            const float cy2 = kv.second.y + kv.second.h / 2;
            const float dx = cx1 - cx2;
            const float dy = cy1 - cy2;
            const float dist = std::sqrt(dx * dx + dy * dy);
            const float scale = std::max({track.w, track.h, kv.second.w, kv.second.h});
            if (scale <= 0.0f) continue;
            if (dist <= stitch_dist_ratio_ * scale && (best < 0 || dist < best_dist)) {
                best = cid;
                best_dist = dist;
            }
        }
        const int canonical = (best >= 0) ? best : raw;
        id_remap_[raw] = canonical;
        tracks[p.first].track_id = canonical;
        taken.insert(canonical);
        raw_last_frame_[raw] = frame_id;
        if (best >= 0) {
            LOG_INFO_FMT("[TrackerNode] Stitch track {} -> {} (dist={:.1f})", raw, canonical, best_dist);
        }
    }

    // 更新归并后轨迹的最新位置
    for (const auto& track : tracks) {
        auto& mem = track_memory_[track.track_id];
        mem.x = track.x;
        mem.y = track.y;
        mem.w = track.w;
        mem.h = track.h;
        mem.last_frame_id = frame_id;
    }
}

void TrackerNode::cleanupStitchState(int64_t frame_id) {
    const int64_t keep = frame_id - stitch_memory_frames_;
    for (auto it = raw_last_frame_.begin(); it != raw_last_frame_.end();) {
        if (it->second < keep) {
            id_remap_.erase(it->first);
            it = raw_last_frame_.erase(it);
        } else {
            ++it;
        }
    }
    for (auto it = track_memory_.begin(); it != track_memory_.end();) {
        if (it->second.last_frame_id < keep) {
            it = track_memory_.erase(it);
        } else {
            ++it;
        }
    }
}

float TrackerNode::computeIoU(const core::InferenceResultPacket::BBox& det,
                               const UnifiedTrackResult& track) {
    float ix1 = std::max(det.x, track.x);
    float iy1 = std::max(det.y, track.y);
    float ix2 = std::min(det.x + det.w, track.x + track.w);
    float iy2 = std::min(det.y + det.h, track.y + track.h);

    if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;

    float intersection = (ix2 - ix1) * (iy2 - iy1);
    float area_det = det.w * det.h;
    float area_track = track.w * track.h;
    return intersection / (area_det + area_track - intersection);
}

// 注册节点
REGISTER_NODE("tracker", TrackerNode)

} // namespace nodes
} // namespace ai_stream