// src/hal/tensorrt/tensorrt_inference_engine.cpp
// TensorRT 推理引擎后端——封装现有 TensorRT 逻辑到 HAL 接口
#include "tensorrt_inference_engine.h"
#include "ai_stream/hal/inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"

#include <cuda_runtime_api.h>

namespace ai_stream {
namespace hal {

TensorrtInferenceEngine::TensorrtInferenceEngine() {
    LOG_DEBUG("[TensorrtInferenceEngine] Constructor");
}

TensorrtInferenceEngine::~TensorrtInferenceEngine() {
    LOG_DEBUG("[TensorrtInferenceEngine] Destructor");
}

bool TensorrtInferenceEngine::loadModel(const InferenceConfig& config) {
    config_ = config;

    if (cudaSetDevice(config.device_id) != cudaSuccess) {
        LOG_ERROR_FMT("[TensorrtInferenceEngine] Cannot select CUDA device {}", config.device_id);
        return false;
    }

    if (!core_.loadEngine(config.model_path, "TensorrtInferenceEngine")) {
        return false;
    }

    // 确定 IO tensor 名称（单输入/单输出模型，取第一个）
    bool output_found = false;
    for (const auto& meta : core_.tensors()) {
        if (meta.is_input) {
            input_name_ = meta.name;
        } else if (!output_found) {
            output_name_ = meta.name;
            output_found = true;
        }
    }

    // 按引擎推导大小分配输入/输出缓冲区并绑定
    for (const auto& meta : core_.tensors()) {
        size_t bytes = meta.elementsIgnoringDynamic() * sizeof(float);
        void* ptr = core_.allocBuffer(meta.name, bytes);
        if (!ptr) return false;
        if (meta.is_input) input_size_ = bytes;
        else output_size_ = bytes;
    }

    core_.setStream(cuda_stream_);
    loaded_ = true;
    LOG_INFO_FMT("[TensorrtInferenceEngine] Model loaded: {}", config.model_path);
    return true;
}

bool TensorrtInferenceEngine::infer(const void* input_data, size_t input_size,
                                     void* output_data, size_t output_size) {
    if (!loaded_) {
        LOG_ERROR("[TensorrtInferenceEngine] Model not loaded");
        return false;
    }
    if (input_size > input_size_) input_size = input_size_;
    if (output_size > output_size_) output_size = output_size_;

    void* d_input = core_.allocBuffer(input_name_, input_size);
    void* d_output = core_.allocBuffer(output_name_, output_size);

    // 拷贝输入到 GPU
    cudaMemcpyAsync(d_input, input_data, input_size, cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(cuda_stream_));

    // 执行推理
    if (!core_.enqueue()) {
        LOG_ERROR("[TensorrtInferenceEngine] enqueueV3 failed");
        return false;
    }

    // 拷贝输出到主机
    cudaMemcpyAsync(output_data, d_output, output_size, cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(cuda_stream_));
    cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_));

    return true;
}

std::pair<int, int> TensorrtInferenceEngine::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int TensorrtInferenceEngine::getBatchSize() const {
    return config_.batch_size;
}

bool TensorrtInferenceEngine::isAvailable() const {
#ifdef WITH_TENSORRT
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

void TensorrtInferenceEngine::setCudaStream(void* stream) {
    cuda_stream_ = stream;
    core_.setStream(stream);
}

void* TensorrtInferenceEngine::getRawContext() const {
    return core_.context();
}

// 注册 TensorRT 后端到工厂
#ifdef WITH_TENSORRT
REGISTER_INFERENCE_BACKEND(InferenceBackend::TENSORRT, TensorrtInferenceEngine)
#endif

} // namespace hal
} // namespace ai_stream
