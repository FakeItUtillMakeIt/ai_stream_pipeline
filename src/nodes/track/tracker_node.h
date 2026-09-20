// src/nodes/track/tracker_node.h
#pragma once

#include "ai_stream/nodes/i_tracker_node.h"
#include "ai_stream/core/queued_node.h"
#include "base_tracker.h"
#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace ai_stream {
namespace nodes {

// 前向声明配置结构
struct OCSortConfig;
struct ByteTrackConfig;

class TrackerNode : public core::QueuedNode<ITrackerNode> {
public:
    TrackerNode();
    ~TrackerNode() override;

    // ITrackerNode 接口
    void setTrackerType(TrackerType type) override;
    TrackerType getTrackerType() const override;
    void setSubStreamId(const std::string& stream_id) override;
    void setTrackerId(const std::string& id) override;
    void setOCSortConfig(const OCSortConfig& config) override;
    void setByteTrackConfig(const ByteTrackConfig& config) override;
    int getActiveTrackCount() const override;

    // QueuedNode 接口
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;
    bool onStartup() override;
    void onShutdown() override;

private:
    bool configureImpl(const std::string& node_id, const nlohmann::json& params) override;

    static float computeIoU(const core::InferenceResultPacket::BBox& det,
                            const UnifiedTrackResult& track);

    // 轨迹 ID 缝合：目标短暂丢失或类别跃迁（如跌倒）导致 tracker 新建 ID 时，
    // 依据时空邻近性将新 ID 归并到最近丢失的轨迹，保持下游 ID 稳定
    void applyStitching(std::vector<UnifiedTrackResult>& tracks, int64_t frame_id);
    void cleanupStitchState(int64_t frame_id);

    struct TrackMemory {
        float x = 0, y = 0, w = 0, h = 0;
        int64_t last_frame_id = -1;
    };

    TrackerType tracker_type_ = TrackerType::OCSORT;
    OCSortConfig ocsort_config_;
    std::string sub_stream_id_;
    std::string tracker_id_;
    ByteTrackConfig bytetrack_config_;
    TrackerPtr tracker_;

    // track_id -> 类别绑定（轨迹诞生时按 IoU 绑定；仅当 name 与 class_id 同时变化时才允许跃迁）。
    // 用于按名称匹配：多推理源融合场景下不同模型的 class_id 可能冲突，
    // 而轨迹身份与语义类别绑定才正确；同时支持同类模型内 person->down 等真实类别跃迁
    struct TrackClassBinding {
        int class_id = -1;
        std::string name;
    };
    std::unordered_map<int, TrackClassBinding> track_class_names_;

    // ===== 轨迹 ID 缝合 =====
    bool stitch_enabled_ = true;
    int stitch_gap_frames_ = 15;        // 允许缝合的最大丢失帧数
    float stitch_dist_ratio_ = 0.5f;    // 中心距离 <= ratio * 最大边长 才缝合
    int64_t stitch_memory_frames_ = 300; // 缝合记忆保留帧数（用于清理）
    int64_t stitch_frame_counter_ = 0;
    std::unordered_map<int, int> id_remap_;              // tracker 原始 id -> 归并后 id
    std::unordered_map<int, int64_t> raw_last_frame_;    // 原始 id -> 最近出现帧
    std::unordered_map<int, TrackMemory> track_memory_;  // 归并后 id -> 最近位置


    // 用于清理过期轨迹的 ID 集合
    std::unordered_set<int> active_track_ids_;

    static constexpr size_t MAX_TRACK_CLASS_NAMES = 500;
};

} // namespace nodes
} // namespace ai_stream