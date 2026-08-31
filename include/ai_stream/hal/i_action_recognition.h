// include/ai_stream/hal/i_action_recognition.h
// 动作识别抽象接口——隔离 TensorRT / RKNN / Ascend OM 等后端
#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief 动作识别结果
 */
struct ActionResult {
    int action_id = -1;
    std::string action_label;
    float confidence = 0.0f;
    std::vector<float> scores;                         // 全类别概率（softmax 后）
    std::vector<std::pair<std::string, float>> top_k;  // top-k 预测
};

/**
 * @brief 动作识别配置
 */
struct ActionRecognitionConfig {
    std::string model_path;
    int input_height = 224;
    int input_width = 224;
    int num_frames = 16;       // clip 帧数
    int frame_interval = 2;    // 帧采样间隔
    int batch_size = 1;
    std::string precision = "fp16";
    std::vector<std::string> action_labels;
    int device_id = 0;
};

/**
 * @brief 动作识别引擎抽象接口
 *
 * 封装基于视频的动作识别模型推理（如 VideoMAE, SlowFast, TSM 等）。
 * 与 IInferenceEngine 不同，动作识别需要处理时序信息（多帧 clip）。
 */
class IActionRecognitionEngine {
public:
    virtual ~IActionRecognitionEngine() = default;

    /**
     * @brief 加载模型
     */
    virtual bool loadModel(const ActionRecognitionConfig& config) = 0;

    /**
     * @brief 执行动作识别推理
     * @param clip_data 输入 clip 数据（RGB，NCHW 布局，num_frames * 3 * H * W）
     * @param clip_size 输入数据大小（字节）
     * @param result 输出结果
     * @return 是否成功
     */
    virtual bool infer(const uint8_t* clip_data, size_t clip_size,
                       ActionResult& result) = 0;

    /**
     * @brief 推理已预处理的 NCHW float 数据（上游预处理节点已完成 resize+normalize）
     * @return 是否支持并成功；后端不支持时返回 false
     */
    virtual bool inferPreprocessed(const float* input_nchw, size_t size_bytes,
                                   ActionResult& result) {
        (void)input_nchw; (void)size_bytes; (void)result;
        return false;
    }

    /**
     * @brief 推理 GPU 帧 clip（每帧为设备端已预处理 NCHW float 数据的指针）
     * @return 是否支持并成功；后端不支持时返回 false
     */
    virtual bool inferGpuFrames(const std::vector<void*>& gpu_frames,
                                ActionResult& result) {
        (void)gpu_frames; (void)result;
        return false;
    }

    /**
     * @brief 设置外部 CUDA 流（以 void* 传递，避免接口泄漏 CUDA 头文件）
     */
    virtual void setCudaStream(void* stream) { (void)stream; }

    /**
     * @brief 获取输入尺寸
     */
    virtual std::pair<int, int> getInputSize() const = 0;

    /**
     * @brief 获取 clip 帧数
     */
    virtual int getNumFrames() const = 0;

    /**
     * @brief 获取引擎名称
     */
    virtual std::string getBackendName() const = 0;

    /**
     * @brief 检查是否可用
     */
    virtual bool isAvailable() const = 0;
};

using ActionRecognitionEnginePtr = std::unique_ptr<IActionRecognitionEngine>;

} // namespace hal
} // namespace ai_stream
