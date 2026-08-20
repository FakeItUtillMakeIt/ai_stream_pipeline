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
    static float computeIoU(const core::InferenceResultPacket::BBox& det,
                            const UnifiedTrackResult& track);

    TrackerType tracker_type_ = TrackerType::OCSORT;
    OCSortConfig ocsort_config_;
    std::string sub_stream_id_;
    std::string tracker_id_;
    ByteTrackConfig bytetrack_config_;
    TrackerPtr tracker_;

    // track_id -> class_name 绑定（轨迹诞生时按 IoU 绑定，终身不变）。
    // 用于按名称匹配：多推理源融合场景下不同模型的 class_id 可能冲突，
    // 而轨迹身份与语义类别绑定才正确
    std::unordered_map<int, std::string> track_class_names_;

    // 用于清理过期轨迹的 ID 集合
    std::unordered_set<int> active_track_ids_;

    static constexpr size_t MAX_TRACK_CLASS_NAMES = 500;
};

} // namespace nodes
} // namespace ai_stream