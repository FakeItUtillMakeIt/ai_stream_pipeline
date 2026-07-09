// src/nodes/fusion/fusion_node.h
#pragma once

#include "ai_stream/nodes/i_fusion_node.h"

namespace ai_stream {
namespace nodes {

    class FusionNodeImpl : public IFusionNode {
    public:
        FusionNodeImpl();
        ~FusionNodeImpl() override = default;

        bool start() override;
        void stop() override;
        void pushData(std::shared_ptr<core::BasePacket> packet) override;
        void setFusionMode(FusionMode mode) override;
        FusionMode getFusionMode() const override;
        void setActionSource(const std::string& source_node_id) override;
        std::string getActionSource() const override;
        void setTimestampThreshold(int64_t threshold_ms) override;
        int64_t getTimestampThreshold() const override;
    private:
        void handleActionResult(std::shared_ptr<core::InferenceResultPacket> result);
        void handleDetectionResult(std::shared_ptr<core::InferenceResultPacket> result);

        FusionMode fusion_mode_ = FusionMode::FRAME_LEVEL;
        std::string action_source_;
        int64_t timestamp_threshold_ms_ = 500;  // 默认500ms阈值

        // 动作识别结果缓存
        core::InferenceResultPacket::ActionResult cached_action_;
        int64_t cached_action_timestamp_ = 0;
        bool has_cached_action_ = false;

        mutable std::mutex mutex_;
    };

} // namespace nodes
} // namespace ai_stream
