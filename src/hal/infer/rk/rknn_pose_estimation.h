// src/hal/infer/rk/rknn_pose_estimation.h
// RKNN 姿态估计引擎——Rockchip RK3588 NPU
//
// 输出契约与 TensorrtPoseEstimation 一致，供 pose 节点 decodeFrame 直接消费：
//   output_host  float [num_persons, 8400, 56]（行主序，逐 anchor 连续）
//   每行 56 = 4 box(未使用) + 1 person score + 51 kpt(x,y,conf)
//   坐标与 score 均为模型输入（640）空间，box 由节点忽略、kpt 按 letterbox
//   参数反解回原图。
//
// 支持的模型输出形态（loadModel 时按 tensor 属性自动识别）：
//   [1, 56, N]  通道优先（YOLOv8-pose 已解码导出），引擎内转置为 [N, 56]
//   [1, N, 56]  行主序，直接拷贝
//   N 必须为 8400（80² + 40² + 20²，与节点固定候选数契约一致）。
//
// 通过 dlopen 加载 librknnrt.so，与 rknn_inference_engine / rknn_detection_engine
// 保持一致，允许在无 NPU 的编译主机构建（构建期仅需 rknn_api.h）。
#pragma once

#include "ai_stream/hal/i_pose_estimation.h"
#include "3rd_party/rk_platform/rknn/include/rknn_api.h"

#include <mutex>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class RknnPoseEstimation : public IPoseEstimationEngine {
public:
    RknnPoseEstimation();
    ~RknnPoseEstimation() override;

    bool loadModel(const PoseEstimationConfig& config) override;

    // host 输入：NCHW float [num_persons, 3, H, W]，逐人单帧推理，
    // 输出拼装为 [num_persons, 8400, 56]。
    bool inferHost(const float* input_nchw, int num_persons,
                   std::vector<float>& output_host) override;

    // NPU 无设备端原图 / 设备端输入路径，返回 false 由节点回退 host 预处理。
    bool inferFromDeviceImage(const void* d_src_bgr,
                              int src_w, int src_h, size_t src_pitch,
                              const std::vector<float>& boxes_7,
                              int num_persons,
                              std::vector<float>& output_host) override {
        (void)d_src_bgr; (void)src_w; (void)src_h; (void)src_pitch;
        (void)boxes_7; (void)num_persons; (void)output_host;
        return false;
    }

    size_t getOutputFloatsPerPerson() const override;
    std::pair<int, int> getInputSize() const override;
    std::string getBackendName() const override { return "RKNN Pose (Rockchip NPU)"; }
    bool isAvailable() const override;

private:
    // 单人推理：输入 NCHW float [1,3,H,W]，输出 [num_anchors_, 56] 写入 out_person
    bool inferOne(const float* input_nchw, float* out_person);

    PoseEstimationConfig config_;
    bool loaded_ = false;

    rknn_context rknn_ctx_ = 0;
    uint32_t n_outputs_ = 1;

    // 输入属性
    int input_width_ = 640;
    int input_height_ = 640;
    rknn_tensor_type input_type_ = RKNN_TENSOR_FLOAT32;
    rknn_tensor_format input_fmt_ = RKNN_TENSOR_NCHW;
    bool input_need_denorm_ = false;  // 模型输入 UINT8/INT8 时需反归一化 + 重排

    // 输出布局
    bool out_row_major_ = false;  // true: [N,56] 直接拷贝；false: [56,N] 需转置
    int num_anchors_ = 0;         // 候选数（契约要求 8400）

    // 反归一化暂存（UINT8 输入路径）
    std::vector<uint8_t> input_nhwc_;

    std::mutex mutex_;
};

} // namespace hal
} // namespace ai_stream