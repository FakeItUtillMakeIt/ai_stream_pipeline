// src/hal/tensorrt/tensorrt_pose_kernels.cu
// TensorRT 姿态估计 GPU 预处理 kernel
#include "tensorrt_pose_kernels.cuh"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ai_stream {
namespace hal {

namespace {

// CUDA Kernel：Letterbox 裁剪 + 等比缩放 + 灰色Padding + Normalize + HWC→NCHW
__global__ void cropResizeNormalizeLetterboxKernel(
    const unsigned char* __restrict__ src,
    int src_w, int src_h, size_t src_pitch,
    float* __restrict__ dst,
    const float* __restrict__ boxes,
    int num_persons,
    int dst_w, int dst_h,
    float mean_b, float mean_g, float mean_r,
    float std_b, float std_g, float std_r,
    float pad_val)
{
    int pw = blockIdx.x * blockDim.x + threadIdx.x;
    int ph = blockIdx.y * blockDim.y + threadIdx.y;
    int p  = blockIdx.z;

    if (p >= num_persons || pw >= dst_w || ph >= dst_h) {
        return;
    }

    float box_x1 = boxes[p * 7 + 0];
    float box_y1 = boxes[p * 7 + 1];
    float box_x2 = boxes[p * 7 + 2];
    float box_y2 = boxes[p * 7 + 3];
    float scale  = boxes[p * 7 + 4];
    float pad_x  = boxes[p * 7 + 5];
    float pad_y  = boxes[p * 7 + 6];

    float content_x_f = pw - pad_x;
    float content_y_f = ph - pad_y;

    float content_w_f = dst_w - 2.0f * pad_x;
    float content_h_f = dst_h - 2.0f * pad_y;

    float b, g, r;

    if (content_x_f < 0.0f || content_x_f >= content_w_f ||
        content_y_f < 0.0f || content_y_f >= content_h_f) {
        b = pad_val;
        g = pad_val;
        r = pad_val;
    } else {
        float src_x = box_x1 + (content_x_f + 0.5f) / scale - 0.5f;
        float src_y = box_y1 + (content_y_f + 0.5f) / scale - 0.5f;

        src_x = fmaxf(0.0f, fminf(src_x, src_w - 1.0f));
        src_y = fmaxf(0.0f, fminf(src_y, src_h - 1.0f));

        int x0 = static_cast<int>(floorf(src_x));
        int y0 = static_cast<int>(floorf(src_y));
        int x0p1 = x0 + 1;
        int y0p1 = y0 + 1;
        int x1_i = (x0p1 < src_w) ? x0p1 : (src_w - 1);
        int y1_i = (y0p1 < src_h) ? y0p1 : (src_h - 1);

        float dx = src_x - x0;
        float dy = src_y - y0;
        float w00 = (1.0f - dx) * (1.0f - dy);
        float w01 = dx * (1.0f - dy);
        float w10 = (1.0f - dx) * dy;
        float w11 = dx * dy;

        const unsigned char* row0 = src + y0 * src_pitch;
        const unsigned char* row1 = src + y1_i * src_pitch;

        b = w00 * row0[x0 * 3 + 0] + w01 * row0[x1_i * 3 + 0]
          + w10 * row1[x0 * 3 + 0] + w11 * row1[x1_i * 3 + 0];
        g = w00 * row0[x0 * 3 + 1] + w01 * row0[x1_i * 3 + 1]
          + w10 * row1[x0 * 3 + 1] + w11 * row1[x1_i * 3 + 1];
        r = w00 * row0[x0 * 3 + 2] + w01 * row0[x1_i * 3 + 2]
          + w10 * row1[x0 * 3 + 2] + w11 * row1[x1_i * 3 + 2];
    }

    b = (b / 255.0f - mean_b) / std_b;
    g = (g / 255.0f - mean_g) / std_g;
    r = (r / 255.0f - mean_r) / std_r;

    int hw = dst_w * dst_h;
    int base = p * 3 * hw + ph * dst_w + pw;
    dst[base + 0 * hw] = b;
    dst[base + 1 * hw] = g;
    dst[base + 2 * hw] = r;
}

} // anonymous namespace

void launchCropResizeNormalizeLetterbox(
    const unsigned char* d_src, int src_w, int src_h, size_t src_pitch,
    float* d_dst,
    const float* d_boxes,
    int num_persons,
    int dst_w, int dst_h,
    float mean_b, float mean_g, float mean_r,
    float std_b, float std_g, float std_r,
    float pad_val,
    void* stream)
{
    dim3 block_size(16, 16, 1);
    dim3 grid_size((dst_w + block_size.x - 1) / block_size.x,
                   (dst_h + block_size.y - 1) / block_size.y,
                   num_persons);

    cropResizeNormalizeLetterboxKernel<<<grid_size, block_size, 0,
        static_cast<cudaStream_t>(stream)>>>(
        d_src, src_w, src_h, src_pitch,
        d_dst, d_boxes, num_persons,
        dst_w, dst_h,
        mean_b, mean_g, mean_r,
        std_b, std_g, std_r,
        pad_val);
}

} // namespace hal
} // namespace ai_stream
