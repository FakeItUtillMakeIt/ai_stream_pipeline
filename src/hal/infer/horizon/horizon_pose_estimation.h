// src/hal/infer/horizon/horizon_pose_estimation.h
// Horizon BPU 姿态估计引擎——地平线 RDK S100P
#pragma once

#include "ai_stream/hal/i_pose_estimation.h"
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class HorizonPoseEstimationEngine : public IPoseEstimationEngine {
public:
    HorizonPoseEstimationEngine() = default;
    ~HorizonPoseEstimationEngine() override;

    bool loadModel(const PoseEstimationConfig& config) override;
    bool inferHost(const float* input_nchw, int num_persons,
                   std::vector<float>& output_host) override;
    std::pair<int, int> getInputSize() const override;
    std::string getBackendName() const override { return "Horizon BPU Pose (RDK)"; }
    bool isAvailable() const override;
    size_t getOutputFloatsPerPerson() const override;

private:
    PoseEstimationConfig config_;
    bool loaded_ = false;
    void* dnn_packed_handle_ = nullptr;
    void* dnn_handle_ = nullptr;
    int input_width_ = 0;
    int input_height_ = 0;
    int output_floats_per_person_ = 0;
};

} // namespace hal
} // namespace ai_stream
