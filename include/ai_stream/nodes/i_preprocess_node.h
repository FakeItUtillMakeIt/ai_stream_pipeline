// include/ai_stream/nodes/i_preprocess_node.h
#pragma once

#include "ai_stream/core/node.h"
#include <string>
#include <vector>
#include <opencv2/core/mat.hpp>

namespace ai_stream {
namespace nodes {

/**
 * @brief 预处理节点接口
 *
 * 负责图像预处理操作：resize、归一化、颜色空间转换等。
 * 接收解码后的图像帧，输出处理后的图像数据，供推理节点使用。
 */
class IPreprocessNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 设置目标尺寸
     * @param width 目标宽度
     * @param height 目标高度
     */
    virtual void setTargetSize(int width, int height) = 0;

    /**
     * @brief 获取当前目标尺寸
     * @return 目标尺寸对 [宽, 高]
     */
    virtual std::pair<int, int> getTargetSize() const = 0;

    /**
     * @brief 设置归一化均值
     * @param mean 均值向量，通常格式为 [B, G, R] 或 [R, G, B]
     */
    virtual void setMean(const std::vector<float>& mean) = 0;

    /**
     * @brief 设置归一化标准差
     * @param std 标准差向量，通常格式为 [B, G, R] 或 [R, G, B]
     */
    virtual void setStd(const std::vector<float>& std) = 0;

    /**
     * @brief 设置插值方法
     * @param method 插值方法字符串，如 "bilinear", "nearest", "cubic"
     */
    virtual void setInterpolationMethod(const std::string& method) = 0;

    /**
     * @brief 设置是否保持宽高比
     * @param keep_aspect_ratio true 表示保持宽高比，会进行填充
     */
    virtual void setKeepAspectRatio(bool keep_aspect_ratio) = 0;

    /**
     * @brief 设置输出数据类型
     * @param dtype 数据类型字符串，如 "float32", "uint8"
     */
    virtual void setOutputDataType(const std::string& dtype) = 0;

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        (void)node_id;
        setTargetSize(params.value("output_width", 640), params.value("output_height", 640));
        if (params.contains("keep_aspect_ratio")) {
            setKeepAspectRatio(params["keep_aspect_ratio"].get<bool>());
        }
        if (params.contains("mean")) {
            setMean(params["mean"].get<std::vector<float>>());
        }
        if (params.contains("std")) {
            setStd(params["std"].get<std::vector<float>>());
        }
        return true;
    }
};

} // namespace nodes
} // namespace ai_stream