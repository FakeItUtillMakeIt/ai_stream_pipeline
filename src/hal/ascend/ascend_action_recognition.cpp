// src/hal/ascend/ascend_action_recognition.cpp
// Ascend OM 动作识别引擎——华为 Ascend NPU
#include "ascend_action_recognition.h"
#include "ai_stream/hal/action_recognition_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

AscendActionRecognition::AscendActionRecognition() {
    LOG_DEBUG("[AscendActionRecognition] Constructor");
}

AscendActionRecognition::~AscendActionRecognition() {
    // 实际实现：释放 ACL 资源
    LOG_DEBUG("[AscendActionRecognition] Destructor");
}

bool AscendActionRecognition::loadModel(const ActionRecognitionConfig& config) {
    config_ = config;
    LOG_INFO_FMT("[AscendActionRecognition] Loading model: {}", config.model_path);

    // 实际实现：
    // 1. aclInit()
    // 2. aclrtSetDevice()
    // 3. aclrtMalloc(&model_work_, work_size)
    // 4. 读取 .om 文件
    // 5. aclmdlLoadFromMem(model_data, model_size, &model_id_)
    // 6. aclmdlCreateDesc(&model_desc_)
    // 7. aclmdlGetDesc(model_desc_, model_id_)
    // 8. 创建输入输出 dataset

    loaded_ = true;
    LOG_INFO_FMT("[AscendActionRecognition] Model loaded (placeholder): {}", config.model_path);
    return true;
}

bool AscendActionRecognition::infer(const uint8_t* clip_data, size_t clip_size,
                                      ActionResult& result) {
    if (!loaded_) {
        LOG_ERROR("[AscendActionRecognition] Model not loaded");
        return false;
    }

    // 实际实现：
    // 1. 预处理 clip 数据
    // 2. aclmdlExecute(model_id_, input_dataset_, output_dataset_)
    // 3. 后处理获取动作类别

    // 占位：返回默认结果
    if (!config_.action_labels.empty()) {
        result.action_id = 0;
        result.action_label = config_.action_labels[0];
        result.confidence = 0.5f;
    }

    LOG_DEBUG_FMT("[AscendActionRecognition] Inference done (placeholder)");
    return true;
}

std::pair<int, int> AscendActionRecognition::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int AscendActionRecognition::getNumFrames() const {
    return config_.num_frames;
}

bool AscendActionRecognition::isAvailable() const {
#ifdef WITH_ASCEND
    return false;
#else
    return false;
#endif
}

bool AscendActionRecognition::initAcl() {
    LOG_DEBUG("[AscendActionRecognition] initAcl (placeholder)");
    return true;
}

bool AscendActionRecognition::preprocessClip(const uint8_t* clip_data, void* device_input) {
    // 实际实现：DVPP 预处理或 CPU 预处理后拷贝到设备
    return true;
}

ActionResult AscendActionRecognition::postprocess(const void* device_output, int num_classes) {
    ActionResult result;
    // 实际实现：softmax + argmax
    return result;
}

// 注册 Ascend 动作识别后端到工厂
#ifdef WITH_ASCEND
REGISTER_ACTION_RECOGNITION_BACKEND(ActionRecognitionBackend::ASCEND, AscendActionRecognition)
#endif

} // namespace hal
} // namespace ai_stream
