// include/ai_stream/nodes/i_action_recognition_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "ai_stream/core/packet.h"
#include <string>
#include <vector>

namespace ai_stream {
namespace nodes {

/**
 * @brief 动作识别节点接口
 * 
 * 支持VideoMAE等视频理解模型，通过帧缓冲和滑动窗口进行动作识别
 */
class IActionRecognitionNode : public core::Node {
public:
    using core::Node::Node;
    virtual ~IActionRecognitionNode() = default;
    
    /**
     * @brief 设置模型路径
     * @param model_path 模型文件路径（.onnx 或 .engine）
     */
    virtual void setModelPath(const std::string& model_path) = 0;
    
    /**
     * @brief 设置输入视频尺寸
     * @param height 输入帧高度
     * @param width 输入帧宽度
     */
    virtual void setInputSize(int height, int width) = 0;
    
    /**
     * @brief 设置clip参数
     * @param num_frames 每个clip的帧数（如16帧）
     * @param frame_interval 帧采样间隔（如每隔2帧取1帧）
     */
    virtual void setClipParams(int num_frames, int frame_interval = 1) = 0;
    
    /**
     * @brief 设置滑动窗口参数
     * @param window_size 窗口大小（帧数）
     * @param stride 窗口滑动步长（帧数）
     */
    virtual void setSlidingWindow(int window_size, int stride = 8) = 0;
    
    /**
     * @brief 设置动作标签
     * @param labels 动作类别标签列表
     */
    virtual void setActionLabels(const std::vector<std::string>& labels) = 0;
    
    /**
     * @brief 设置置信度阈值
     * @param threshold 置信度阈值
     */
    virtual void setConfidenceThreshold(float threshold) = 0;
    
    /**
     * @brief 设置批处理大小
     * @param batch_size 批处理大小
     */
    virtual void setBatchSize(int batch_size) = 0;

    virtual void setDeviceId(int device_id) { (void)device_id; }

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        (void)node_id;
        if (params.contains("input_height") && params.contains("input_width")) {
            setInputSize(params["input_height"].get<int>(), params["input_width"].get<int>());
        }
        if (params.contains("num_frames") && params.contains("frame_interval")) {
            setClipParams(params["num_frames"].get<int>(), params["frame_interval"].get<int>());
        }
        if (params.contains("window_size") && params.contains("stride")) {
            setSlidingWindow(params["window_size"].get<int>(), params["stride"].get<int>());
        }
        if (params.contains("action_labels") && params["action_labels"].is_array()) {
            setActionLabels(params["action_labels"].get<std::vector<std::string>>());
        }
        if (params.contains("confidence_threshold")) {
            setConfidenceThreshold(params["confidence_threshold"].get<float>());
        }
        if (params.contains("batch_size")) {
            setBatchSize(params["batch_size"].get<int>());
        }
        if (params.contains("device_id")) {
            setDeviceId(params["device_id"].get<int>());
        }
        if (params.contains("model_path")) {
            setModelPath(params["model_path"].get<std::string>());
        }
        return true;
    }
};

} // namespace nodes
} // namespace ai_stream
