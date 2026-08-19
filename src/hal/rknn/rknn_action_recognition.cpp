// src/hal/rknn/rknn_action_recognition.cpp
// RKNN 动作识别引擎——Rockchip RK3588 NPU
#include "rknn_action_recognition.h"
#include "ai_stream/hal/action_recognition_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

RknnActionRecognition::RknnActionRecognition() {
    LOG_DEBUG("[RknnActionRecognition] Constructor");
}

RknnActionRecognition::~RknnActionRecognition() {
    // 实际实现：rknn_destroy(rknn_ctx_);
    LOG_DEBUG("[RknnActionRecognition] Destructor");
}

bool RknnActionRecognition::loadModel(const ActionRecognitionConfig& config) {
    config_ = config;
    LOG_INFO_FMT("[RknnActionRecognition] Loading model: {}", config.model_path);

    // 实际实现：
    // 1. 读取 .rknn 文件到内存
    // 2. rknn_init(&rknn_ctx_, model_data, model_size, 0, nullptr)
    // 3. 查询输入输出属性
    // 4. 对于时序模型，可能需要特殊处理（分帧推理或模型拆分）

    loaded_ = true;
    LOG_INFO_FMT("[RknnActionRecognition] Model loaded (placeholder): {}", config.model_path);
    return true;
}

bool RknnActionRecognition::infer(const uint8_t* clip_data, size_t clip_size,
                                    ActionResult& result) {
    if (!loaded_) {
        LOG_ERROR("[RknnActionRecognition] Model not loaded");
        return false;
    }

    // 实际实现：
    // 1. 预处理 clip 数据（resize, normalize）
    // 2. rknn_inputs_set(rknn_ctx_, 1, &input)
    // 3. rknn_run(rknn_ctx_, nullptr)
    // 4. rknn_outputs_get(rknn_ctx_, 1, &output, nullptr)
    // 5. 后处理获取动作类别

    // 占位：返回默认结果
    if (!config_.action_labels.empty()) {
        result.action_id = 0;
        result.action_label = config_.action_labels[0];
        result.confidence = 0.5f;
    }

    LOG_DEBUG_FMT("[RknnActionRecognition] Inference done (placeholder)");
    return true;
}

std::pair<int, int> RknnActionRecognition::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int RknnActionRecognition::getNumFrames() const {
    return config_.num_frames;
}

bool RknnActionRecognition::isAvailable() const {
#ifdef WITH_RKNN
    return true;
#else
    return false;
#endif
}

bool RknnActionRecognition::initRknnContext() {
    LOG_DEBUG("[RknnActionRecognition] initRknnContext (placeholder)");
    return true;
}

bool RknnActionRecognition::preprocessClip(const uint8_t* clip_data, float* input_buffer) {
    // 实际实现：对每帧进行 resize、归一化
    return true;
}

ActionResult RknnActionRecognition::postprocess(const float* output, int num_classes) {
    ActionResult result;
    // 实际实现：softmax + argmax
    return result;
}

// 注册 RKNN 动作识别后端到工厂
#ifdef WITH_RKNN
REGISTER_ACTION_RECOGNITION_BACKEND(ActionRecognitionBackend::RKNN, RknnActionRecognition)
#endif

} // namespace hal
} // namespace ai_stream
