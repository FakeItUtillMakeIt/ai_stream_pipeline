// src/nodes/fusion/fusion_node.h
#pragma once

#include "ai_stream/nodes/i_fusion_node.h"
#include "ai_stream/core/queued_node.h"
#include <map>
#include <set>
#include <deque>
#include <unordered_map>
#include <vector>

namespace ai_stream {
namespace nodes {

    class FusionNodeImpl : public core::QueuedNode<IFusionNode> {
    public:
        FusionNodeImpl();
        ~FusionNodeImpl() override { stop(); }

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
        // 动作合流(frame_level)时丢弃非推理结果的原始帧透传。
        // VideoMAE 节点会把每个输入原始帧也 broadcast 下来（保持帧流不断），
        // 但这些 DECODED_FRAME 经 alert 透传到 evidence/sink 会变成"无框帧"，
        // 与 draw 的有框帧交替 → 证据视频/预览闪烁。开启后 fusion 只放行
        // InferenceResultPacket，丢弃这些原始帧。默认 false 保持既有行为。
        bool drop_non_inference_ = false;
        // 动作合流(frame_level)时把最近一次动作持续附加到每帧，直到下次推理覆盖。
        // 用于 VideoMAE 这类隔窗才出结果的模型：否则规则侧只见到孤立单帧动作，
        // 攒不够 duration_ms 就永不告警。默认 false 保持"一次性消费"旧行为。
        bool action_persist_ = false;

        // 帧配对状态（仅 worker 线程访问）
        struct PendingFrame {
            std::unordered_map<std::string, std::shared_ptr<core::InferenceResultPacket>> per_source;
            int64_t first_arrival_ms = 0;
        };
        std::map<std::pair<uint32_t, int64_t>, PendingFrame> pending_;
        static constexpr size_t MAX_PENDING_FRAMES = 1000;

        // 已吐出去的帧键（stream_id, frame_id）。DETECTION_MERGE 下，某帧一旦因
        // "配齐"或"超时半包"广播过，就记入此集合；之后迟到的另一路结果直接丢弃，
        // 不再新建 pending 二次广播。否则同一源帧会被吐两帧（一帧有框、一帧没框），
        // 下游录像按 pts 逐帧写 → 证据视频"一顿一顿 + 框闪烁"。
        // 用 set 快速查 + deque 做有界淘汰（只保留最近 MAX_EMITTED_KEYS 帧）。
        std::set<std::pair<uint32_t, int64_t>> emitted_keys_;
        std::deque<std::pair<uint32_t, int64_t>> emitted_fifo_;
        static constexpr size_t MAX_EMITTED_KEYS = 4096;

        bool wasEmitted(const std::pair<uint32_t, int64_t>& key) const
        {
            return emitted_keys_.count(key) > 0;
        }
        void markEmitted(const std::pair<uint32_t, int64_t>& key)
        {
            if (emitted_keys_.insert(key).second) {
                emitted_fifo_.push_back(key);
                while (emitted_fifo_.size() > MAX_EMITTED_KEYS) {
                    emitted_keys_.erase(emitted_fifo_.front());
                    emitted_fifo_.pop_front();
                }
            }
        }
    };

} // namespace nodes
} // namespace ai_stream
