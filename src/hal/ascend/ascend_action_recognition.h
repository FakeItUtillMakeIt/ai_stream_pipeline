// src/hal/ascend/ascend_action_recognition.h
// Ascend OM 动作识别引擎——华为 Ascend NPU
#pragma once

#include "ai_stream/hal/i_action_recognition.h"
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief Ascend OM 动作识别引擎
 *
 * 使用华为 Ascend OM (Offline Model) 在 Ascend NPU 上执行动作识别推理。
 * 模型需要预先通过 ATC 工具转换为 .om 格式。
 *
 * Ascend 310/910 支持复杂的时序模型推理。
 */
class AscendActionRecognition : public IActionRecognitionEngine {
public:
    AscendActionRecognition();
    ~AscendActionRecognition() override;

    bool loadModel(const ActionRecognitionConfig& config) override;
    bool infer(const uint8_t* clip_data, size_t clip_size,
               ActionResult& result) override;
    std::pair<int, int> getInputSize() const override;
    int getNumFrames() const override;
    std::string getBackendName() const override { return "Ascend OM Action Recognition (Ascend NPU)"; }
    bool isAvailable() const override;

private:
    bool initAcl();
    bool preprocessClip(const uint8_t* clip_data, void* device_input);
    ActionResult postprocess(const void* device_output, int num_classes);

    ActionRecognitionConfig config_;
    bool loaded_ = false;
    void* model_work_ = nullptr;
    void* model_weight_ = nullptr;
    void* model_desc_ = nullptr;
    uint32_t model_id_ = 0;
    void* input_dataset_ = nullptr;
    void* output_dataset_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
