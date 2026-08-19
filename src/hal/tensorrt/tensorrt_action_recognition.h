// src/hal/tensorrt/tensorrt_action_recognition.h
// TensorRT 动作识别引擎——封装现有 VideoMAE 推理逻辑到 HAL 接口
#pragma once

#include "ai_stream/hal/i_action_recognition.h"
#include <string>
#include <memory>

namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace ai_stream {
namespace hal {

/**
 * @brief TensorRT 动作识别引擎
 *
 * 封装 TensorRT 推理逻辑到 IActionRecognitionEngine 接口。
 * 支持 VideoMAE, SlowFast, TSM 等时序模型。
 */
class TensorrtActionRecognition : public IActionRecognitionEngine {
public:
    TensorrtActionRecognition();
    ~TensorrtActionRecognition() override;

    bool loadModel(const ActionRecognitionConfig& config) override;
    bool infer(const uint8_t* clip_data, size_t clip_size,
               ActionResult& result) override;
    std::pair<int, int> getInputSize() const override;
    int getNumFrames() const override;
    std::string getBackendName() const override { return "TensorRT Action Recognition (NVIDIA)"; }
    bool isAvailable() const override;

    void setCudaStream(void* stream);

private:
    bool initEngine(const std::string& engine_path);
    bool allocateBuffers();
    void freeBuffers();
    void preprocessClip(const uint8_t* clip_data, float* gpu_input);
    ActionResult postprocess(const float* output, int num_classes);

    ActionRecognitionConfig config_;
    bool loaded_ = false;

    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_{
        nullptr, [](nvinfer1::IRuntime* p){ if (p) delete p; }};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_{
        nullptr, [](nvinfer1::ICudaEngine* p){ if (p) delete p; }};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_{
        nullptr, [](nvinfer1::IExecutionContext* p){ if (p) delete p; }};

    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    size_t input_size_ = 0;
    size_t output_size_ = 0;
    void* cuda_stream_ = nullptr;

    std::string input_name_ = "input";
    std::string output_name_ = "output";
};

} // namespace hal
} // namespace ai_stream
