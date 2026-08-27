// src/hal/tensorrt/tensorrt_pose_kernels.cuh
// TensorRT 姿态估计 GPU 预处理 kernel 启动器（.cu 实现，供 .cpp 调用）
#pragma once

#include <cstddef>

namespace ai_stream {
namespace hal {

/**
 * @brief 启动融合预处理 kernel：Letterbox 裁剪 + 等比缩放 + 灰色Padding
 *        + Normalize + HWC→NCHW
 * @param boxes 设备端 [num_persons, 7] = x1,y1,x2,y2,scale,pad_x,pad_y
 */
void launchCropResizeNormalizeLetterbox(
    const unsigned char* d_src, int src_w, int src_h, size_t src_pitch,
    float* d_dst,
    const float* d_boxes,
    int num_persons,
    int dst_w, int dst_h,
    float mean_b, float mean_g, float mean_r,
    float std_b, float std_g, float std_r,
    float pad_val,
    void* stream);

} // namespace hal
} // namespace ai_stream
