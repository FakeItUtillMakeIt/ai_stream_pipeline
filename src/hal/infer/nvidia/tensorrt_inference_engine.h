// src/hal/tensorrt/tensorrt_inference_engine.h
// TensorRT 推理引擎后端——封装现有 TensorRT 逻辑到 HAL 接口
#pragma once

#include "ai_stream/hal/i_inference_engine.h"
#include "trt_core.h"
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief TensorRT 通用推理引擎（字节级进出，任意模型兜底抽象）
 *
 * 引擎生命周期与缓冲区管理由 TrtCore 内核承担，
 * 本类只负责 IInferenceEngine 契约的适配。
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
    InferenceConfig config_;
    bool loaded_ = false;

    // 公共内核：引擎生命周期 / 缓冲区 / 执行
    TrtCore core_;

    // CUDA 流
    void* cuda_stream_ = nullptr;

    // IO 字节大小
    size_t input_size_ = 0;
    size_t output_size_ = 0;

    // Tensor 名称
    std::string input_name_ = "images";
    std::string output_name_ = "output0";
};

} // namespace hal
} // namespace ai_stream
