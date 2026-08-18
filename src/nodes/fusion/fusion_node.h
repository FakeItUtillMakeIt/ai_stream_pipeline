// src/nodes/fusion/fusion_node.h
#pragma once

#include "ai_stream/nodes/i_fusion_node.h"
#include "ai_stream/core/queued_node.h"
#include <map>
#include <unordered_map>
#include <vector>

namespace ai_stream {
namespace nodes {

    class FusionNodeImpl : public core::QueuedNode<IFusionNode> {
    public:
        FusionNodeImpl();
        ~FusionNodeImpl() override = default;

        // QueuedNode 接口
        void processPacket(std::shared_ptr<core::BasePacket> packet) override;
        void onIdle() override;
        bool onStartup() override;
        void onShutdown() override;
        bool configureImpl(const std::string& node_id, const nlohmann::json& params) override;

        // IFusionNode 接口
        void setFusionMode(FusionMode mode) override;
        FusionMode getFusionMode() const override;
        void setActionSource(const std::string& source_node_id) override;
        std::string getActionSource() const override;
        void setTimestampThreshold(int64_t threshold_ms) override;
        int64_t getTimestampThreshold() const override;

    private:
        // 动作融合（FRAME_LEVEL/OBJECT_LEVEL 原有逻辑）
        void handleActionResult(std::shared_ptr<core::InferenceResultPacket> result);
        void handleDetectionResult(std::shared_ptr<core::InferenceResultPacket> result);

        // 多推理源检测框融合（DETECTION_MERGE）
        void handleDetectionMerge(std::shared_ptr<core::InferenceResultPacket> result);
        void flushExpiredPending(int64_t now_ms);
        void mergeAndBroadcast(const std::pair<uint32_t, int64_t>& key);
        bool isMergeSource(const std::string& producer) const;

        FusionMode fusion_mode_ = FusionMode::FRAME_LEVEL;
        std::string action_source_;
        int64_t timestamp_threshold_ms_ = 500;  // 默认500ms阈值

        // 动作识别结果缓存
        core::InferenceResultPacket::ActionResult cached_action_;
        int64_t cached_action_timestamp_ = 0;
        bool has_cached_action_ = false;

        // ===== DETECTION_MERGE 配置 =====
        std::vector<std::string> merge_sources_;                 // 参与融合的节点 id（producer_id）
        std::unordered_map<std::string, int> class_offsets_;     // 节点 id -> class_id 偏移（可选）
        int64_t wait_timeout_ms_ = 200;                          // 帧配对超时（超时广播部分合并）
        bool cross_nms_ = false;                                 // 跨源 NMS 开关（默认关）
        float nms_iou_threshold_ = 0.5f;

        // 帧配对状态（仅 worker 线程访问）
        struct PendingFrame {
            std::unordered_map<std::string, std::shared_ptr<core::InferenceResultPacket>> per_source;
            int64_t first_arrival_ms = 0;
        };
        std::map<std::pair<uint32_t, int64_t>, PendingFrame> pending_;
        static constexpr size_t MAX_PENDING_FRAMES = 1000;

        mutable std::mutex mutex_;
    };

} // namespace nodes
} // namespace ai_stream
