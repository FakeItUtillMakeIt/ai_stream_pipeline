// src/nodes/fusion/fusion_node.cpp
#include "fusion_node.h"
#include "utils/time_util.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "registry/node_factory.h"
#include <cstdlib>

namespace ai_stream {
namespace nodes {

class FusionNodeImpl : public IFusionNode {
public:
    FusionNodeImpl() : IFusionNode("Fusion") {}
    ~FusionNodeImpl() override = default;

    bool start() override {
        running_ = true;
        LOG_INFO_FMT("[Fusion] Started with mode: {}", 
                     fusion_mode_ == FusionMode::FRAME_LEVEL ? "FRAME_LEVEL" : "OBJECT_LEVEL");
        return true;
    }

    void stop() override {
        running_ = false;
        LOG_INFO_FMT("[Fusion] Stopped");
    }

    void pushData(std::shared_ptr<core::BasePacket> packet) override {
        if (!running_) return;

        auto result = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
        if (!result) {
            // 非推理结果包，直接转发
            broadcast(packet);
            return;
        }

        std::lock_guard<std::mutex> lock(mutex_);

        // 判断数据来源
        if (!result->action_results.empty()) {
            // 来自action_recognition：更新缓存
            handleActionResult(result);
        } 
        else if (!result->detections.empty()) {
            // 来自tracker：融合动作结果并广播
            handleDetectionResult(result);
        }
    }

    // IFusionNode接口实现
    void setFusionMode(FusionMode mode) override {
        fusion_mode_ = mode;
    }

    FusionMode getFusionMode() const override {
        return fusion_mode_;
    }

    void setActionSource(const std::string& source_node_id) override {
        action_source_ = source_node_id;
    }

    std::string getActionSource() const override {
        return action_source_;
    }

    void setTimestampThreshold(int64_t threshold_ms) override {
        timestamp_threshold_ms_ = threshold_ms;
    }

    int64_t getTimestampThreshold() const override {
        return timestamp_threshold_ms_;
    }

private:
    void handleActionResult(std::shared_ptr<core::InferenceResultPacket> result) {
        // 缓存最新的动作识别结果
        cached_action_ = result->action_results[0];
        cached_action_timestamp_ = result->timestamp_ms;
        has_cached_action_ = true;

        LOG_DEBUG_FMT("[Fusion] Cached action: {} (confidence: {:.4f}, timestamp: {}ms)",
                     cached_action_.action_label, cached_action_.confidence, cached_action_timestamp_);
    }

    void handleDetectionResult(std::shared_ptr<core::InferenceResultPacket> result) {
        // 检查是否有缓存的动作结果
        if (!has_cached_action_) {
            // 没有动作识别结果，直接转发检测结果
            broadcast(result);
            return;
        }

        // 检查时间戳是否在阈值范围内
        int64_t time_diff = std::abs(result->timestamp_ms - cached_action_timestamp_);
        if (time_diff > timestamp_threshold_ms_) {
            // 动作识别结果太旧，不融合
            LOG_DEBUG_FMT("[Fusion] Action result too old (diff: {}ms > threshold: {}ms), skipping",
                         time_diff, timestamp_threshold_ms_);
            broadcast(result);
            return;
        }

        // 融合：将动作结果附加到检测结果
        result->action_results.push_back(cached_action_);

        LOG_DEBUG_FMT("[Fusion] Fused detection with action: {} (time_diff: {}ms)",
                     cached_action_.action_label, time_diff);

        // 广播融合后的结果
        broadcast(result);
    }

    // 融合配置
    FusionMode fusion_mode_ = FusionMode::FRAME_LEVEL;
    std::string action_source_;
    int64_t timestamp_threshold_ms_ = 500;  // 默认500ms阈值

    // 动作识别结果缓存
    core::InferenceResultPacket::ActionResult cached_action_;
    int64_t cached_action_timestamp_ = 0;
    bool has_cached_action_ = false;

    mutable std::mutex mutex_;
};

// 注册节点
REGISTER_NODE("fusion", FusionNodeImpl)

} // namespace nodes
} // namespace ai_stream
