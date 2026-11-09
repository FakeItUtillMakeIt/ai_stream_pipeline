// src/hal/infer/ascend/ascend_action_recognition.h
// Ascend OM 动作识别引擎——华为 Ascend NPU
#pragma once

#include "ai_stream/hal/i_action_recognition.h"

#include <acl/acl.h>

#include <cstdint>
#include <string>

namespace ai_stream {
namespace hal {

class AscendActionRecognition : public IActionRecognitionEngine {
public:
    AscendActionRecognition() = default;
    ~AscendActionRecognition() override;

    bool loadModel(const ActionRecognitionConfig& config) override;
    bool infer(const uint8_t* clip_data, size_t clip_size,
               ActionResult& result) override;
    bool inferPreprocessed(const float* input_nchw, size_t size_bytes,
                           ActionResult& result) override;

    std::pair<int, int> getInputSize() const override;
    int getNumFrames() const override;
    std::string getBackendName() const override;
    bool isAvailable() const override;

private:
    bool initAcl();
    bool parseIo();
    ActionResult postprocess(const float* logits, size_t count) const;
    void cleanup();

    ActionRecognitionConfig config_;
    std::string soc_name_;
    bool loaded_ = false;

    uint32_t model_id_ = 0;
    aclmdlDesc* desc_ = nullptr;
    aclmdlDataset* input_ds_ = nullptr;
    aclmdlDataset* output_ds_ = nullptr;
    void* d_input_ = nullptr;
    size_t input_size_ = 0;
    aclDataBuffer* input_buf_ = nullptr;
    void* d_output_ = nullptr;
    size_t output_size_ = 0;
    size_t output_elements_ = 0;
    aclDataBuffer* output_buf_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
