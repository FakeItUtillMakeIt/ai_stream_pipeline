// src/nodes/infer/action_recognition_videomae.h
#pragma once

#include "ai_stream/nodes/i_action_recognition_node.h"
#include "ai_stream/core/queued_node.h"
#include "ai_stream/hal/i_action_recognition.h"
#include "ai_stream/hal/action_recognition_factory.h"
#include <opencv2/opencv.hpp>
#include <deque>
#include <unordered_map>
#include <memory>
#include <mutex>

namespace ai_stream {
namespace nodes {

/**
 * @brief VideoMAE动作识别节点
 * 
 * 使用轻量化VideoMAE模型进行动作识别
 * 支持帧缓冲和滑动窗口机制
 */
class ActionRecognitionVideoMAENode : public core::QueuedNode<IActionRecognitionNode> {
public:
    struct Config {
        std::string model_path;           // 模型路径
        int input_height = 224;           // 输入高度
        int input_width = 224;            // 输入宽度
        int num_frames = 16;              // clip帧数
        int frame_interval = 2;           // 帧采样间隔
        int window_size = 16;             // 滑动窗口大小
        int stride = 8;                   // 滑动步长
        float confidence_threshold = 0.5f;
        int batch_size = 1;
        std::vector<std::string> action_labels;
    };
    
    ActionRecognitionVideoMAENode();
    explicit ActionRecognitionVideoMAENode(const Config& cfg);
    ~ActionRecognitionVideoMAENode();

    // QueuedNode接口实现
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;
    bool onStartup() override;
    void onShutdown() override;
    
    // IActionRecognitionNode接口实现
    void setModelPath(const std::string& model_path) override;
    void setInputSize(int height, int width) override;
    void setClipParams(int num_frames, int frame_interval) override;
    void setSlidingWindow(int window_size, int stride) override;
    void setActionLabels(const std::vector<std::string>& labels) override;
    void setConfidenceThreshold(float threshold) override;
    void setBatchSize(int batch_size) override;

private:
    // 推理（通过 HAL 后端引擎）
    bool loadModel();
    void inferFromGpu(const std::vector<void*>& gpu_frames, core::InferenceResultPacket::ActionResult& result);
    void inferFromCpu(const std::vector<cv::Mat>& clip, core::InferenceResultPacket::ActionResult& result);

    // 帧缓冲和滑动窗口
    void updateGpuFrameBuffer(uint32_t stream_id, void* d_ptr, int64_t frame_id);
    void updateFrameBuffer(uint32_t stream_id, const cv::Mat& frame, int64_t frame_id);
    bool shouldRunInference(uint32_t stream_id);
    std::vector<void*> getGpuClipFromBuffer(uint32_t stream_id);
    std::vector<cv::Mat> getClipFromBuffer(uint32_t stream_id);

    // 预处理（CPU路径使用：HWC float → NCHW float 布局转换）
    void preprocessClip(const std::vector<cv::Mat>& clip, float* input_buffer);

    // hal::ActionResult → core::ActionResult 映射
    core::InferenceResultPacket::ActionResult toCoreResult(const hal::ActionResult& hal_result) const;

    Config cfg_;

    // HAL 推理引擎后端（TensorRT/RKNN/Ascend 由工厂按编译配置选择）
    hal::ActionRecognitionEnginePtr engine_;

    // CPU帧缓冲区（用于接收CPU预处理节点的输出）
    struct FrameBuffer {
        std::deque<std::pair<int64_t, cv::Mat>> frames;  // (frame_id, frame)
    };
    std::unordered_map<uint32_t, FrameBuffer> frame_buffers_;
    
    // GPU帧缓冲区（用于接收GPU预处理节点的输出）
    struct GpuFrameBuffer {
        std::deque<std::pair<int64_t, void*>> frames;  // (frame_id, d_ptr)
    };
    std::unordered_map<uint32_t, GpuFrameBuffer> gpu_frame_buffers_;
    
    // 滑动窗口状态
    struct WindowState {
        int frame_counter = 0;
        int last_inference_frame = -1;
    };
    std::unordered_map<uint32_t, WindowState> window_states_;
    
    mutable std::mutex mutex_;
    bool is_initialized_ = false;
};

} // namespace nodes
} // namespace ai_stream
