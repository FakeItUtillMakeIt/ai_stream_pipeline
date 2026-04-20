// src/nodes/track/base_tracker.h
#pragma once

#include "track_result.h"
#include "ai_stream/core/packet.h"
#include "ai_stream/nodes/i_tracker_node.h"
#include <memory>

namespace ai_stream {
namespace nodes {

/**
 * @brief 跟踪器抽象基类
 */
class BaseTracker {
public:
    virtual ~BaseTracker() = default;
    
    /**
     * @brief 获取跟踪器类型
     */
    virtual TrackerType getType() const = 0;
    
    /**
     * @brief 更新跟踪
     * @param detections 检测结果
     * @return 带跟踪 ID 的结果
     */
    virtual std::vector<UnifiedTrackResult> update(
        const std::vector<core::InferenceResultPacket::BBox>& detections) = 0;
    
    /**
     * @brief 获取活跃跟踪数量
     */
    virtual int getActiveCount() const = 0;
    
    /**
     * @brief 重置跟踪器
     */
    virtual void reset() = 0;
};

using TrackerPtr = std::shared_ptr<BaseTracker>;

} // namespace nodes
} // namespace ai_stream