// src/hal/cpu/cpu_inference_engine.h
// CPU 推理引擎——基于 OpenCV DNN 模块的轻量推理后端
#pragma once

#include "ai_stream/hal/i_inference_engine.h"
#include <opencv2/dnn.hpp>

namespace ai_stream {
namespace hal {

class CpuInferenceEngine : public IInferenceEngine {
public:
    CpuInferenceEngine() = default;
    ~CpuInferenceEngine() override = default;

    bool loadModel(const InferenceConfig& config) override;
    bool infer(const void* input_data, size_t input_size,
               void* output_data, size_t output_size) override;
    std::pair<int, int> getInputSize() const override;
    int getBatchSize() const override;
    std::string getBackendName() const override { return "CPU (OpenCV DNN)"; }
    bool isAvailable() const override;

private:
    InferenceConfig config_;
    bool loaded_ = false;
    cv::dnn::Net net_;
};

} // namespace hal
} // namespace ai_stream
