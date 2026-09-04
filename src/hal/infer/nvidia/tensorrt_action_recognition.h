// src/hal/tensorrt/tensorrt_action_recognition.h
// TensorRT 动作识别引擎——封装现有 VideoMAE 推理逻辑到 HAL 接口
#pragma once

#include "ai_stream/hal/i_action_recognition.h"
#include "trt_core.h"
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief TensorRT 动作识别引擎
 *
 * 封装 TensorRT 推理逻辑到 IActionRecognitionEngine 接口。
 * 支持 VideoMAE, SlowFast, TSM 等时序模型。
 * 引擎生命周期与缓冲区由 TrtCore 内核承担。
 */
class TensorrtActionRecognition : public IActionRecognitionEngine {
public:
    TensorrtActionRecognition();
    ~TensorrtActionRecognition() override;

    bool loadModel(const ActionRecognitionConfig& config) override;
    bool infer(const uint8_t* clip_data, size_t clip_size,
               ActionResult& result) override;
    bool inferPreprocessed(const float* input_nchw, size_t size_bytes,
                           ActionResult& result) override;
    bool inferGpuFrames(const std::vector<void*>& gpu_frames,
                        ActionResult& result) override;
    void setCudaStream(void* stream) override;
    std::pair<int, int> getInputSize() const override;
    int getNumFrames() const override;
    std::string getBackendName() const override { return "TensorRT Action Recognition (NVIDIA)"; }
    bool isAvailable() const override;

private:
    void preprocessClip(const uint8_t* clip_data, float* gpu_input);
    // 设置动态输入形状并执行推理 + D2H 拷贝 + 后处理
    bool enqueueAndPostprocess(ActionResult& result);
    ActionResult postprocess(const float* output, int num_classes);

    ActionRecognitionConfig config_;
    bool loaded_ = false;

    // 公共内核：引擎生命周期 / 缓冲区 / 执行
    TrtCore core_;

    std::string input_name_ = "input";
    std::string output_name_ = "output";
};

} // namespace hal
} // namespace ai_stream
