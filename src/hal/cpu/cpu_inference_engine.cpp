// src/hal/cpu/cpu_inference_engine.cpp
// CPU 推理引擎——基于 OpenCV DNN 模块的轻量推理后端
#include "cpu_inference_engine.h"
#include "ai_stream/hal/inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <cstring>

namespace ai_stream {
namespace hal {

bool CpuInferenceEngine::loadModel(const InferenceConfig& config) {
    config_ = config;
    try {
        net_ = cv::dnn::readNet(config.model_path);
        loaded_ = !net_.empty();
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[CpuInferenceEngine] Failed to load model '{}': {}", config.model_path, e.what());
        loaded_ = false;
    }
    return loaded_;
}

bool CpuInferenceEngine::infer(const void* input_data, size_t input_size,
                                void* output_data, size_t output_size) {
    if (!loaded_) {
        LOG_ERROR("[CpuInferenceEngine] Model not loaded");
        return false;
    }
    if (!input_data || !output_data || input_size == 0 || output_size == 0) return false;

    const size_t image_size = static_cast<size_t>(3) * config_.input_width *
                              config_.input_height * sizeof(float);
    if (image_size == 0 || input_size % image_size != 0) return false;

    const int batch = static_cast<int>(input_size / image_size);
    int shape[] = {batch, 3, config_.input_height, config_.input_width};
    cv::Mat input(4, shape, CV_32F, const_cast<void*>(input_data));
    net_.setInput(input, config_.input_name);

    cv::Mat output = config_.output_name.empty()
                         ? net_.forward()
                         : net_.forward(config_.output_name);
    const size_t result_size = output.total() * output.elemSize();
    if (result_size == 0 || result_size > output_size) return false;
    std::memcpy(output_data, output.ptr(), result_size);
    return true;
}

std::pair<int, int> CpuInferenceEngine::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int CpuInferenceEngine::getBatchSize() const {
    return config_.batch_size;
}

bool CpuInferenceEngine::isAvailable() const {
    // CPU 后端始终可用（作为最后 fallback）
    return true;
}

// 注册 CPU 后端到工厂
REGISTER_INFERENCE_BACKEND(InferenceBackend::CPU, CpuInferenceEngine)

} // namespace hal
} // namespace ai_stream
