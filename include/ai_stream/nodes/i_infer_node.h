// include/ai_stream/nodes/i_infer_node.h
#pragma once

#include "ai_stream/core/node.h"
#include <string>
#include <vector>

namespace ai_stream {
namespace nodes {

/**
 * @brief 检测器类型枚举
 *
 * 定义支持的检测器类型，用于区分不同的推理任务
 */
enum class DetectorType {
    DETECTION = 0,      /// 目标检测（框+类别）
    SEGMENTATION = 1,   /// 实例分割
    CLASSIFICATION = 2  /// 图像分类
};

/**
 * @brief 推理节点接口
 *
 * 接收解码后的图像帧，执行深度学习模型推理，输出元数据（如检测框、分类结果）。
 */
class IInferNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 加载模型文件
     * @param model_path 模型文件路径（如 .engine、.onnx）
     * @return 是否加载成功
     */
    virtual bool loadModel(const std::string& model_path) = 0;

    /**
     * @brief 设置推理精度（FP32/FP16/INT8）
     * @param precision 精度字符串
     */
    virtual void setPrecision(const std::string& precision) = 0;

    /**
     * @brief 设置批处理大小
     * @param batch_size 批大小，>1 时启用动态批处理
     */
    virtual void setBatchSize(int batch_size) = 0;

    /**
     * @brief 获取当前模型输入尺寸要求 [宽, 高]
     */
    virtual std::pair<int, int> getInputSize() const = 0;

    /**
     * @brief 获取模型类别名称列表
     */
    virtual std::vector<std::string> getClassNames() const = 0;

    /**
     * @brief 设置检测器类型
     * @param type 检测器类型枚举值
     */
    virtual void setDetectorType(DetectorType type) = 0;

    /**
     * @brief 获取当前检测器类型
     * @return 检测器类型枚举值
     */
    virtual DetectorType getDetectorType() const = 0;
};

} // namespace nodes
} // namespace ai_stream