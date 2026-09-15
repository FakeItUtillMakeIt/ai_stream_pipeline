// src/hal/infer/horizon/horizon_inference_engine.h
// Horizon BPU 推理引擎后端——地平线 RDK S100P
#pragma once

#include "ai_stream/hal/i_inference_engine.h"
#include <string>
#include <vector>

// Forward declare Horizon SDK types to avoid header leakage
typedef void* hbDNNPackedHandle_t;
typedef void* hbDNNHandle_t;
typedef void* hbUCPTaskHandle_t;

namespace ai_stream {
namespace hal {

/**
 * @brief Horizon BPU 推理引擎
 *
 * 使用地平线 DNN SDK 在 BPU 上执行推理。
 * 模型需要预先通过地平线工具链转换为 .bin/.json 或 .hbm 格式。
 */
class HorizonInferenceEngine : public IInferenceEngine {
public:
    HorizonInferenceEngine();
    ~HorizonInferenceEngine() override;

    bool loadModel(const InferenceConfig& config) override;
    bool infer(const void* input_data, size_t input_size,
               void* output_data, size_t output_size) override;
    std::pair<int, int> getInputSize() const override;
    int getBatchSize() const override;
    std::string getBackendName() const override { return "Horizon BPU (RDK)"; }
    bool isAvailable() const override;

private:
    bool loadDnnLib();
    bool queryModelInfo();

    InferenceConfig config_;
    bool loaded_ = false;

    // Horizon DNN handles
    hbDNNPackedHandle_t dnn_packed_handle_ = nullptr;
    hbDNNHandle_t dnn_handle_ = nullptr;

    // Model info
    int input_width_ = 0;
    int input_height_ = 0;
    int num_inputs_ = 0;
    int num_outputs_ = 0;
    std::vector<int64_t> output_sizes_;

    // dlopen handle
    void* dl_handle_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
