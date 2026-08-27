// src/hal/tensorrt/tensorrt_pose_estimation.h
// TensorRT 姿态估计引擎——封装现有 YOLO-Pose 推理逻辑到 HAL 接口
#pragma once

#include "ai_stream/hal/i_pose_estimation.h"
#include "trt_core.h"
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

/**
 * @brief TensorRT 姿态估计引擎
 *
 * 支持 YOLO-Pose 类模型（输出 [batch, 8400, 56] 转置格式）。
 * 三种推理路径全部实现：host 输入 / 设备端输入 / 设备端原图（GPU 融合预处理 kernel）。
 * 引擎生命周期与缓冲区由 TrtCore 内核承担。
 */
class TensorrtPoseEstimation : public IPoseEstimationEngine {
public:
    TensorrtPoseEstimation();
    ~TensorrtPoseEstimation() override;

    bool loadModel(const PoseEstimationConfig& config) override;
    bool inferHost(const float* input_nchw, int num_persons,
                   std::vector<float>& output_host) override;
    bool inferDevice(const void* d_input_nchw, int num_persons,
                     std::vector<float>& output_host) override;
    bool inferFromDeviceImage(const void* d_src_bgr,
                              int src_w, int src_h, size_t src_pitch,
                              const std::vector<float>& boxes_7,
                              int num_persons,
                              std::vector<float>& output_host) override;
    void setCudaStream(void* stream) override;

    size_t getOutputFloatsPerPerson() const override;
    std::pair<int, int> getInputSize() const override;
    std::string getBackendName() const override { return "TensorRT Pose Estimation (NVIDIA)"; }
    bool isAvailable() const override;

private:
    // 设置动态 batch、绑定地址并执行推理，拷贝输出到 host
    bool enqueueAndFetch(int num_persons, const void* d_input,
                         std::vector<float>& output_host);

    PoseEstimationConfig config_;
    bool loaded_ = false;

    // 公共内核：引擎生命周期 / 缓冲区 / 执行
    TrtCore core_;

    // 模型输入尺寸
    int input_w_ = 640;
    int input_h_ = 640;

    static constexpr int NUM_CANDIDATES = 8400;  // 80*80 + 40*40 + 20*20
    static constexpr int POSE_DIM = 56;           // 4 box + 1 score + 51 kpts

    std::string input_name_ = "images";
    std::string output_name_ = "output0_transposed";
};

} // namespace hal
} // namespace ai_stream
