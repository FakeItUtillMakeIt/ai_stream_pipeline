// include/ai_stream/nodes/i_fusion_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "ai_stream/core/packet.h"
#include <string>
#include <mutex>
#include <unordered_map>

namespace ai_stream {
namespace nodes {

/**
 * @brief 融合模式枚举
 */
enum class FusionMode {
    FRAME_LEVEL,      // 帧级别融合：动作识别结果附加到整个帧
    OBJECT_LEVEL,     // 目标级别融合：动作识别结果关联到特定目标（需要track_id）
    DETECTION_MERGE   // 多推理源检测框融合：按 (stream_id, frame_id) 配对合并多路检测结果
};

/**
 * @brief 融合节点接口
 * 
 * 将多个推理节点的输出结果融合成一个统一的数据包：
 * - 检测框（来自tracker）
 * - 动作识别结果（来自VideoMAE）
 * 
 * 输出的InferenceResultPacket同时包含检测结果和动作识别结果，
 * 下游节点分别处理这两类数据。
 */
class IFusionNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 设置融合模式
     * @param mode 融合模式
     */
    virtual void setFusionMode(FusionMode mode) = 0;

    /**
     * @brief 获取融合模式
     * @return 当前融合模式
     */
    virtual FusionMode getFusionMode() const = 0;

    /**
     * @brief 设置动作识别结果的源节点ID
     * @param source_node_id 动作识别节点的ID
     */
    virtual void setActionSource(const std::string& source_node_id) = 0;

    /**
     * @brief 获取动作识别结果的源节点ID
     * @return 动作识别节点的ID
     */
    virtual std::string getActionSource() const = 0;

    /**
     * @brief 设置时间戳匹配阈值（毫秒）
     * @param threshold_ms 时间阈值，超过此时间差的动作结果不融合
     */
    virtual void setTimestampThreshold(int64_t threshold_ms) = 0;

    /**
     * @brief 获取时间戳匹配阈值
     * @return 时间阈值（毫秒）
     */
    virtual int64_t getTimestampThreshold() const = 0;
};

} // namespace nodes
} // namespace ai_stream
