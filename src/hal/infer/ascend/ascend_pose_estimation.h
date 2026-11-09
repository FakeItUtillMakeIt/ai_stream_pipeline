// src/hal/infer/ascend/ascend_pose_estimation.h
// Ascend ACL YOLO-Pose 姿态估计——输出 [N, 8400, 56]
#pragma once

#include "ai_stream/hal/i_pose_estimation.h"

#include <acl/acl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class AscendPoseEstimation : public IPoseEstimationEngine {
public:
    AscendPoseEstimation() = default;
    ~AscendPoseEstimation() override;

    bool loadModel(const PoseEstimationConfig& config) override;
    bool inferHost(const float* input_nchw, int num_persons,
                   std::vector<float>& output_host) override;

    size_t getOutputFloatsPerPerson() const override { return 8400ULL * 56ULL; }
    std::pair<int, int> getInputSize() const override { return {input_w_, input_h_}; }
    std::string getBackendName() const override;
    bool isAvailable() const override;

private:
    bool initAcl();
    bool parseIo();
    void cleanup();

    PoseEstimationConfig config_;
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
    aclDataBuffer* output_buf_ = nullptr;

    int input_w_ = 640;
    int input_h_ = 640;
    size_t output_floats_ = 8400ULL * 56ULL;
    bool output_needs_transpose_ = false;
};

} // namespace hal
} // namespace ai_stream
