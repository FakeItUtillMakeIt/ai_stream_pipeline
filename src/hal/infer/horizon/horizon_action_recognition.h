// src/hal/infer/horizon/horizon_action_recognition.h
// Horizon BPU 动作识别引擎——地平线 RDK S100P
#pragma once

#include "ai_stream/hal/i_action_recognition.h"
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class HorizonActionRecognitionEngine : public IActionRecognitionEngine {
public:
    HorizonActionRecognitionEngine() = default;
    ~HorizonActionRecognitionEngine() override;

    bool loadModel(const ActionRecognitionConfig& config) override;
    bool infer(const uint8_t* clip_data, size_t clip_size, ActionResult& result) override;
    std::pair<int, int> getInputSize() const override;
    int getNumFrames() const override;
    std::string getBackendName() const override { return "Horizon BPU ActionRec (RDK)"; }
    bool isAvailable() const override;

private:
    ActionRecognitionConfig config_;
    bool loaded_ = false;
    void* dnn_packed_handle_ = nullptr;
    void* dnn_handle_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
