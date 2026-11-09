// src/hal/infer/ascend/ascend_detection_engine.h
// 华为昇腾 ACL 检测推理引擎（Ascend 310P/310/910）
//
// 模型为 ATC 转换后的 .om，输入 1x3xHxW FP32 RGB，输出与 TensorRT 检测后端
// 一致的 5 张量契约：
//   det_boxes(N,4 cxcywh) / det_scores(N) / det_classes(N,1)
//   det_batch_ids(N,1) / det_num_dets()
//
// ACL 的 device context 是线程相关的，infer() 入口会按需 aclrtSetDevice，
// 因此引擎可以在节点 worker 线程里执行。
#pragma once

#include "ai_stream/hal/i_detection_inference_engine.h"

#include <acl/acl.h>

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class AscendDetectionEngine : public IDetectionInferenceEngine {
public:
    AscendDetectionEngine() = default;
    ~AscendDetectionEngine() override;

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
    bool isAvailable() const override;

private:
    struct OutputBuffer {
        std::string name;      // OM 里的完整输出名
        size_t size = 0;       // 字节数
        aclDataType type = ACL_FLOAT;
        void* dev = nullptr;
        aclDataBuffer* data_buffer = nullptr;
        size_t elements = 0;   // 元素个数
    };

    bool initAcl();
    bool parseIo();
    bool copyOutputs();
    bool copyToExternalBuffers();
    void cleanup();

    DetectionInferenceConfig config_;
    int device_id_ = 0;
    std::string soc_name_;
    bool acl_device_set_ = false;

    uint32_t model_id_ = 0;
    bool model_loaded_ = false;
    aclmdlDesc* desc_ = nullptr;
    aclmdlDataset* input_ds_ = nullptr;
    aclmdlDataset* output_ds_ = nullptr;

    void* d_input_ = nullptr;
    size_t input_size_ = 0;
    aclDataBuffer* input_data_buffer_ = nullptr;
    std::vector<OutputBuffer> outputs_;  // 按 OM 输出索引
    std::map<std::string, size_t> output_index_;  // 契约名 -> OM 输出索引

    const void* input_ptr_ = nullptr;
    std::string input_name_ = "images";
    std::map<std::string, void*> external_outputs_;

    std::vector<float> boxes_;
    std::vector<float> scores_;
    std::vector<int64_t> classes_;
    std::vector<int64_t> batch_ids_;
    int64_t num_dets_ = 0;

    std::vector<std::string> input_names_;
    std::vector<std::string> output_names_;
};

} // namespace hal
} // namespace ai_stream
