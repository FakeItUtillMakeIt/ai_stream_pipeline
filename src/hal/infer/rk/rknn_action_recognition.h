// src/hal/rknn/rknn_action_recognition.h
// RKNN 动作识别引擎——Rockchip RK3588 NPU
#pragma once

#include "ai_stream/hal/i_action_recognition.h"
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief RKNN 动作识别引擎
 *
 * 使用 Rockchip RKNN API 在 RK3588 NPU 上执行动作识别推理。
 * 模型需要预先通过 rknn-toolkit2 转换为 .rknn 格式。
 *
 * 注意：RK3588 NPU 对时序模型（VideoMAE, SlowFast）的支持有限，
 * 可能需要将模型拆分为空间特征提取（NPU）和时间融合（CPU）。
 */
class RknnActionRecognition : public IActionRecognitionEngine {
public:
    RknnActionRecognition();
    ~RknnActionRecognition() override;

    bool loadModel(const ActionRecognitionConfig& config) override;
    bool infer(const uint8_t* clip_data, size_t clip_size,
               ActionResult& result) override;
    std::pair<int, int> getInputSize() const override;
    int getNumFrames() const override;
    std::string getBackendName() const override { return "RKNN Action Recognition (Rockchip NPU)"; }
    bool isAvailable() const override;

private:
    bool initRknnContext();
    bool preprocessClip(const uint8_t* clip_data, float* input_buffer);
    ActionResult postprocess(const float* output, int num_classes);

    ActionRecognitionConfig config_;
    bool loaded_ = false;
    void* rknn_ctx_ = nullptr;
    int ctx_flags_ = 0;
};

} // namespace hal
} // namespace ai_stream
