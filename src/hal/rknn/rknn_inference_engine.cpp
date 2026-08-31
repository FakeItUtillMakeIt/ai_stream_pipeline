// src/hal/rknn/rknn_inference_engine.cpp
// RKNN 推理引擎后端——Rockchip RK3588 NPU
#include "rknn_inference_engine.h"
#include "ai_stream/hal/inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

// 实际使用时取消注释：
// #include <rknn_api.h>

namespace ai_stream {
namespace hal {

RknnInferenceEngine::RknnInferenceEngine() {
    LOG_DEBUG("[RknnInferenceEngine] Constructor");
}

RknnInferenceEngine::~RknnInferenceEngine() {
    // 实际实现：rknn_destroy(rknn_ctx_);
    LOG_DEBUG("[RknnInferenceEngine] Destructor");
}

bool RknnInferenceEngine::loadModel(const InferenceConfig& config) {
    config_ = config;
    LOG_INFO_FMT("[RknnInferenceEngine] Loading model: {}", config.model_path);

    // 实际实现：
    // 1. 读取 .rknn 文件到内存
    // 2. rknn_init(&rknn_ctx_, model_data, model_size, 0, nullptr)
    // 3. queryTensorAttr() 获取输入输出属性
    // 4. 设置 core_mask（如果指定）

    // 占位：模拟加载成功
    loaded_ = true;
    LOG_INFO_FMT("[RknnInferenceEngine] Model loaded (placeholder): {}", config.model_path);
    return true;
}

bool RknnInferenceEngine::infer(const void* input_data, size_t input_size,
                                 void* output_data, size_t output_size) {
    if (!loaded_) {
        LOG_ERROR("[RknnInferenceEngine] Model not loaded");
        return false;
    }

    // 实际实现：
    // 1. rknn_inputs_set(rknn_ctx_, 1, &input)
    // 2. rknn_run(rknn_ctx_, nullptr)
    // 3. rknn_outputs_get(rknn_ctx_, 1, &output, nullptr)
    // 4. 拷贝输出到 output_data

    // 占位：将输入拷贝到输出
    size_t copy_size = std::min(input_size, output_size);
    if (input_data && output_data && copy_size > 0) {
        std::memcpy(output_data, input_data, copy_size);
    }
    return true;
}

std::pair<int, int> RknnInferenceEngine::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int RknnInferenceEngine::getBatchSize() const {
    return config_.batch_size;
}

bool RknnInferenceEngine::isAvailable() const {
    // 实际实现：检查 /dev/rknpu 设备是否存在
    // 或尝试加载 rknn_api.so
#ifdef WITH_RKNN
    return false;
#else
    return false;
#endif
}

void RknnInferenceEngine::setCoreMask(int core_mask) {
    core_mask_ = core_mask;
}

bool RknnInferenceEngine::initRknnContext() {
    // 实际实现：rknn_init
    return true;
}

bool RknnInferenceEngine::queryTensorAttr() {
    // 实际实现：rknn_query for input/output tensor attributes
    return true;
}

bool RknnInferenceEngine::setInputs(const void* input_data, size_t input_size) {
    // 实际实现：rknn_inputs_set
    return true;
}

bool RknnInferenceEngine::getOutputs(void* output_data, size_t output_size) {
    // 实际实现：rknn_outputs_get
    return true;
}

// 注册 RKNN 后端到工厂
#ifdef WITH_RKNN
REGISTER_INFERENCE_BACKEND(InferenceBackend::RKNN, RknnInferenceEngine)
#endif

} // namespace hal
} // namespace ai_stream
