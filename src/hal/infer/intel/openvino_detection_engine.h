// src/hal/infer/intel/openvino_detection_engine.h
// Intel OpenVINO 检测推理引擎（CPU / iGPU / NPU）
//
// 使用 OpenVINO C API（openvino/c/openvino.h）而不是 C++ API：
// C ABI 不受 libstdc++ 的 _GLIBCXX_USE_CXX11_ABI 影响，可直接链接
// pip wheel 内的 libopenvino_c，也能链接官方 Runtime。
//
// 输出契约与 TensorRT/RKNN/Horizon 检测后端一致（5 个张量）：
//   det_boxes (N,4) cxcywh | det_scores (N) | det_classes (N,1)
//   det_batch_ids (N,1) | det_num_dets ()
#pragma once

#include "ai_stream/hal/i_detection_inference_engine.h"

#include <openvino/c/openvino.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class OpenVinoDetectionEngine : public IDetectionInferenceEngine {
public:
    OpenVinoDetectionEngine() = default;
    ~OpenVinoDetectionEngine() override;

    bool loadModel(const DetectionInferenceConfig& config) override;

    bool setInputTensor(const std::string& name, void* gpu_ptr) override;
    bool setOutputTensor(const std::string& name, void* gpu_ptr) override;
    void* getOutputTensor(const std::string& name) override;
    size_t getOutputTensorSize(const std::string& name) const override;
    bool allocateOutputBuffers() override { return true; }

    bool infer() override;
    bool inferAsync(void* /*stream*/) override { return infer(); }
    bool synchronize(void* /*stream*/) override { return true; }

    std::vector<std::string> getInputNames() const override;
    std::vector<std::string> getOutputNames() const override;
    std::pair<int, int> getInputSize() const override {
        return {config_.input_width, config_.input_height};
    }
    int getMaxBatchSize() const override { return config_.max_batch_size; }
    std::string getBackendName() const override;
    bool isAvailable() const override { return true; }

private:
    void cleanup();
    bool parseOutputIndices();
    bool copyOutputs();
    bool copyToExternalBuffers();

    DetectionInferenceConfig config_;
    std::string device_name_ = "CPU";

    ov_core_t* core_ = nullptr;
    ov_model_t* model_ = nullptr;
    ov_compiled_model_t* compiled_ = nullptr;
    ov_infer_request_t* request_ = nullptr;

    // 输出名 → 张量索引
    std::map<std::string, size_t> output_index_;
    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;

    const void* input_ptr_ = nullptr;
    std::string input_name_ = "images";
    std::map<std::string, void*> external_outputs_;

    // 内部输出缓冲（getOutputTensor 返回其地址）
    std::vector<float> boxes_;
    std::vector<float> scores_;
    std::vector<int64_t> classes_;
    std::vector<int64_t> batch_ids_;
    int64_t num_dets_ = 0;
};

} // namespace hal
} // namespace ai_stream
