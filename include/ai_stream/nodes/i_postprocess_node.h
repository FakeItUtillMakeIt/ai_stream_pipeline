// include/ai_stream/nodes/i_postprocess_node.h
#pragma once

#include "ai_stream/core/node.h"
#include <string>
#include <vector>

namespace ai_stream {
namespace nodes {

/**
 * @brief 后处理节点接口
 *
 * 负责模型推理结果的后处理：NMS、置信度过滤、坐标转换等。
 * 接收推理节点的输出元数据，生成最终的可视化或业务数据。
 */
class IPostprocessNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 设置置信度阈值
     * @param threshold 置信度阈值，范围 [0.0, 1.0]
     */
    virtual void setConfidenceThreshold(float threshold) = 0;

    /**
     * @brief 获取当前置信度阈值
     * @return 置信度阈值
     */
    virtual float getConfidenceThreshold() const = 0;

    /**
     * @brief 设置 NMS 阈值
     * @param threshold NMS 阈值，范围 [0.0, 1.0]
     */
    virtual void setNmsThreshold(float threshold) = 0;

    /**
     * @brief 获取当前 NMS 阈值
     * @return NMS 阈值
     */
    virtual float getNmsThreshold() const = 0;

    /**
     * @brief 设置最大检测数量
     * @param max_detections 每帧最大检测框数量
     */
    virtual void setMaxDetections(int max_detections) = 0;

    /**
     * @brief 设置类别白名单
     * @param class_names 允许输出的类别名称列表
     */
    virtual void setClassWhitelist(const std::vector<std::string>& class_names) = 0;

    /**
     * @brief 设置是否输出跟踪ID（用于多目标跟踪场景）
     * @param enable 是否启用跟踪ID输出
     */
    virtual void setTrackIdEnabled(bool enable) = 0;

    /**
     * @brief 设置后处理类型
     * @param type 后处理类型字符串，如 "detection", "segmentation", "classification"
     */
    virtual void setPostProcessType(const std::string& type) = 0;

    /**
     * @brief 获取当前后处理类型
     * @return 后处理类型字符串
     */
    virtual std::string getPostProcessType() const = 0;
};

} // namespace nodes
} // namespace ai_stream