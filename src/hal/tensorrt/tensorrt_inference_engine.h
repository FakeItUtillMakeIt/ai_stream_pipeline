// src/hal/tensorrt/tensorrt_inference_engine.h
// TensorRT 推理引擎后端——封装现有 TensorRT 逻辑到 HAL 接口
#pragma once

#include "ai_stream/hal/i_inference_engine.h"
#include <vector>
#include <string>
#include <memory>

// 前向声明 TensorRT 类型
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace ai_stream {
namespace hal {

/**
 * @brief TensorRT 推理引擎
 *
 * 封装 TensorRT 推理逻辑到 IInferenceEngine 接口。
 * 使推理节点可以通过统一接口调用不同后端。
 */
class TensorrtInferenceEngine : public IInferenceEngine {
public:
    TensorrtInferenceEngine();
    ~TensorrtInferenceEngine() override;

    bool loadModel(const InferenceConfig& config) override;
    bool infer(const void* input_data, size_t input_size,
               void* output_data, size_t output_size) override;
    std::pair<int, int> getInputSize() const override;
    int getBatchSize() const override;
    std::string getBackendName() const override { return "TensorRT (NVIDIA)"; }
    bool isAvailable() const override;

    /**
     * @brief 设置 CUDA 流
     */
    void setCudaStream(void* stream);

    /**
     * @brief 获取原始 TensorRT context（用于高级优化）
     */
    void* getRawContext() const;

private:
    static void deleteRuntime(nvinfer1::IRuntime* p);
    static void deleteEngine(nvinfer1::ICudaEngine* p);
    static void deleteContext(nvinfer1::IExecutionContext* p);

    bool initEngine(const std::string& engine_path);
    bool allocateBuffers();
    void freeBuffers();

    InferenceConfig config_;
    bool loaded_ = false;

    // TensorRT 资源
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_{
        nullptr, &TensorrtInferenceEngine::deleteRuntime};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_{
        nullptr, &TensorrtInferenceEngine::deleteEngine};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_{
        nullptr, &TensorrtInferenceEngine::deleteContext};

    // GPU 缓冲区
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    size_t input_size_ = 0;
    size_t output_size_ = 0;

    // CUDA 流
    void* cuda_stream_ = nullptr;

    // Tensor 名称
    std::string input_name_ = "images";
    std::string output_name_ = "output0";
};

} // namespace hal
} // namespace ai_stream
