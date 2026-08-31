// include/ai_stream/nodes/i_infer_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "3rd_party/log_mgr/log_mgr.h"
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
    CLASSIFICATION = 2,  /// 图像分类
    POSE = 3
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

    virtual void setDeviceId(int device_id) { (void)device_id; }

    /**
     * @brief 设置模型输入尺寸要求 [宽, 高]
     * @param width 输入宽度
     * @param height 输入高度
     */
    virtual void setInputSize(int width, int height) = 0;

    /**
     * @brief 获取当前模型输入尺寸要求 [宽, 高]
     */
    virtual std::pair<int, int> getInputSize() const = 0;

    /**
     * @brief 设置模型类别名称列表
     * @param names 类别名称列表
     */
    virtual void setClassNames(const std::vector<std::string>& names) = 0;

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

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        (void)node_id;
        if (!params.contains("detector_config")) {
            return true;
        }
        const auto& detector_config = params["detector_config"];
        if (detector_config.contains("input_size")) {
            const auto& input_size = detector_config["input_size"];
            setInputSize(input_size.value("width", 640), input_size.value("height", 640));
        }
        if (detector_config.contains("batch_size")) {
            setBatchSize(detector_config["batch_size"].get<int>());
        }
        if (detector_config.contains("device_id")) {
            setDeviceId(detector_config["device_id"].get<int>());
        }
        if (detector_config.contains("model_path")) {
            std::string model_path = detector_config["model_path"].get<std::string>();
            if (!loadModel(model_path)) {
                LOG_ERROR_FMT("[IInferNode] Failed to load model: {}", model_path);
                return false;
            }
        }
        if (detector_config.contains("model_class_names")) {
            setClassNames(detector_config["model_class_names"].get<std::vector<std::string>>());
        }
        return true;
    }
};

} // namespace nodes
} // namespace ai_stream
