// src/hal/cpu/cpu_inference_engine.cpp
// CPU 推理引擎——基于 OpenCV DNN 模块的轻量推理后端
#include "cpu_inference_engine.h"
#include "ai_stream/hal/inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

bool CpuInferenceEngine::loadModel(const InferenceConfig& config) {
    config_ = config;
    // 注意：完整实现需要 cv::dnn::Net 加载 ONNX 模型
    // 这里仅作为 HAL 层占位，真实 RK3588 / Ascend 后端会替换
    LOG_INFO_FMT("[CpuInferenceEngine] loadModel: {} (placeholder, use RKNN/Ascend for real inference)",
                 config.model_path);
    loaded_ = true;
    return true;
}

bool CpuInferenceEngine::infer(const void* input_data, size_t input_size,
                                void* output_data, size_t output_size) {
    if (!loaded_) {
        LOG_ERROR("[CpuInferenceEngine] Model not loaded");
        return false;
    }
    // 占位：将输入拷贝到输出（无实际推理）
    // 真实实现需要 cv::dnn::blobFromImage + net.forward()
    size_t copy_size = std::min(input_size, output_size);
    if (input_data && output_data && copy_size > 0) {
        std::memcpy(output_data, input_data, copy_size);
    }
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
