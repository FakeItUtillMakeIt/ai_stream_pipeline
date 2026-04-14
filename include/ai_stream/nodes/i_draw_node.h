// include/ai_stream/nodes/i_draw_node.h
#pragma once

#include "ai_stream/core/node.h"
#include <string>
#include <vector>

namespace ai_stream {
namespace nodes {

/**
 * @brief 画框节点接口
 * 
 * 将推理结果（检测框、关键点等）绘制到原始图像帧上，输出带标注的图像。
 */
class IDrawNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 设置边框颜色 (B, G, R)
     */
    virtual void setBoxColor(int b, int g, int r) = 0;

    /**
     * @brief 设置字体厚度
     */
    virtual void setFontThickness(int thickness) = 0;

    /**
     * @brief 设置是否显示置信度
     */
    virtual void setShowConfidence(bool show) = 0;

    /**
     * @brief 设置类别过滤列表（只绘制这些类别的框）
     * @param class_ids 要显示的类别 ID 列表，空表示全部显示
     */
    virtual void setClassFilter(const std::vector<int>& class_ids) = 0;
};

} // namespace nodes
} // namespace ai_stream