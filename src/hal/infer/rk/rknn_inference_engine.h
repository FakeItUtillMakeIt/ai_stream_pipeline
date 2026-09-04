// src/hal/rknn/rknn_inference_engine.h
// RKNN 推理引擎后端——Rockchip RK3588 NPU
#pragma once

#include "ai_stream/hal/i_inference_engine.h"
#include "rknn_api.h"
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief RKNN 推理引擎
 *
 * 使用 Rockchip RKNN API 在 RK3588 NPU 上执行推理。
 * 模型需要预先通过 rknn-toolkit2 转换为 .rknn 格式。
 */
class RknnInferenceEngine : public IInferenceEngine {
public:
    RknnInferenceEngine();
    ~RknnInferenceEngine() override;

    bool loadModel(const InferenceConfig& config) override;
    bool infer(const void* input_data, size_t input_size,
               void* output_data, size_t output_size) override;
    std::pair<int, int> getInputSize() const override;
    int getBatchSize() const override;
    std::string getBackendName() const override { return "RKNN (Rockchip NPU)"; }
    bool isAvailable() const override;

    /**
     * @brief 设置 RKNN 核心掩码（多核 NPU）
     * @param core_mask 核心掩码，如 0x07 表示使用 3 个核心
     */
    void setCoreMask(int core_mask);

private:
    bool initRknnContext();
    bool queryTensorAttr();
    bool setInputs(const void* input_data, size_t input_size);
    bool getOutputs(void* output_data, size_t output_size);

    InferenceConfig config_;
    bool loaded_ = false;

    // RKNN 句柄
    rknn_context rknn_ctx_;
    int ctx_flags_;

    // 输入输出属性
    std::vector<rknn_tensor_attr> input_attrs_;
    std::vector<rknn_tensor_attr> output_attrs_;

    int core_mask_;  // 0 = 自动调度
};

} // namespace hal
} // namespace ai_stream
