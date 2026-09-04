// src/hal/npp/cuda_kernels_wrapper.cu
// CUDA kernel 包装函数——供 NPP 图像加速器调用
// 将 __global__ kernel 封装为 C++ 可调用的函数
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================================
// Resize + Normalize Kernels
// ============================================================

__global__ void resizeNormalizeKernel(
    const unsigned char* __restrict__ src,
    int src_w, int src_h, size_t src_pitch,
    float* __restrict__ dst,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    float scale_x = static_cast<float>(src_w) / dst_w;
    float scale_y = static_cast<float>(src_h) / dst_h;

    float src_x = (dst_x + 0.5f) * scale_x - 0.5f;
    float src_y = (dst_y + 0.5f) * scale_y - 0.5f;

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

    float b = w00 * row0[x0 * 3 + 0] + w01 * row0[x1_i * 3 + 0]
            + w10 * row1[x0 * 3 + 0] + w11 * row1[x1_i * 3 + 0];
    float g = w00 * row0[x0 * 3 + 1] + w01 * row0[x1_i * 3 + 1]
            + w10 * row1[x0 * 3 + 1] + w11 * row1[x1_i * 3 + 1];
    float r = w00 * row0[x0 * 3 + 2] + w01 * row0[x1_i * 3 + 2]
            + w10 * row1[x0 * 3 + 2] + w11 * row1[x1_i * 3 + 2];

    int hw = dst_w * dst_h;
    int idx = dst_y * dst_w + dst_x;
    dst[0 * hw + idx] = (r / 255.0f - mean_r) / std_r;
    dst[1 * hw + idx] = (g / 255.0f - mean_g) / std_g;
    dst[2 * hw + idx] = (b / 255.0f - mean_b) / std_b;
}

__global__ void letterboxResizeNormalizeKernel(
    const unsigned char* __restrict__ src,
    int src_w, int src_h, size_t src_pitch,
    float* __restrict__ dst,
    int dst_w, int dst_h,
    int letter_w, int letter_h,
    int pad_x, int pad_y,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b)
{
    int dst_x = blockIdx.x * blockDim.x + threadIdx.x;
    int dst_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (dst_x >= dst_w || dst_y >= dst_h) return;

    int hw = dst_w * dst_h;
    int idx = dst_y * dst_w + dst_x;

    // 检查是否在 letterbox 区域内
    bool in_letter_region = (dst_x >= pad_x && dst_x < pad_x + letter_w &&
                             dst_y >= pad_y && dst_y < pad_y + letter_h);

    if (!in_letter_region) {
        // Pad region: fill with gray (114/255.0)
        float gray_val = (114.0f / 255.0f);
        dst[0 * hw + idx] = (gray_val - mean_r) / std_r;
        dst[1 * hw + idx] = (gray_val - mean_g) / std_g;
        dst[2 * hw + idx] = (gray_val - mean_b) / std_b;
        return;
    }

    float src_x = (dst_x - pad_x + 0.5f) * static_cast<float>(src_w) / letter_w - 0.5f;
    float src_y = (dst_y - pad_y + 0.5f) * static_cast<float>(src_h) / letter_h - 0.5f;

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

    float b = w00 * row0[x0 * 3 + 0] + w01 * row0[x1_i * 3 + 0]
            + w10 * row1[x0 * 3 + 0] + w11 * row1[x1_i * 3 + 0];
    float g = w00 * row0[x0 * 3 + 1] + w01 * row0[x1_i * 3 + 1]
            + w10 * row1[x0 * 3 + 1] + w11 * row1[x1_i * 3 + 1];
    float r = w00 * row0[x0 * 3 + 2] + w01 * row0[x1_i * 3 + 2]
            + w10 * row1[x0 * 3 + 2] + w11 * row1[x1_i * 3 + 2];

    dst[0 * hw + idx] = (r / 255.0f - mean_r) / std_r;
    dst[1 * hw + idx] = (g / 255.0f - mean_g) / std_g;
    dst[2 * hw + idx] = (b / 255.0f - mean_b) / std_b;
}

// ============================================================
// Draw Rect Kernel
// ============================================================

__global__ void drawRectKernel(
    unsigned char* image, int width, int height, int pitch,
    int x, int y, int w, int h, int thickness,
    unsigned char b, unsigned char g, unsigned char r)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    bool on_border = false;

    if (py >= y && py < y + thickness && px >= x && px < x + w) on_border = true;
    if (py >= y + h - thickness && py < y + h && px >= x && px < x + w) on_border = true;
    if (px >= x && px < x + thickness && py >= y && py < y + h) on_border = true;
    if (px >= x + w - thickness && px < x + w && py >= y && py < y + h) on_border = true;

    if (on_border) {
        int idx = py * pitch + px * 3;
        image[idx + 0] = b;
        image[idx + 1] = g;
        image[idx + 2] = r;
    }
}

// ============================================================
// NV12 → BGR24 Kernel（硬件解码帧颜色转换，BT.601 有限范围）
// ============================================================

__global__ void nv12ToBgrKernel(
    const unsigned char* __restrict__ src_y, int src_pitch_y,
    const unsigned char* __restrict__ src_uv, int src_pitch_uv,
    unsigned char* __restrict__ dst, int dst_pitch,
    int width, int height)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    const int y_val = src_y[py * src_pitch_y + px];

    const int uv_row = py >> 1;
    const int uv_col = px >> 1;
    const size_t uv_idx = static_cast<size_t>(uv_row) * src_pitch_uv + uv_col * 2;
    const int u = src_uv[uv_idx];
    const int v = src_uv[uv_idx + 1];

    // BT.601 有限范围：C = Y-16, D = U-128, E = V-128
    const int c = y_val - 16;
    const int d = u - 128;
    const int e = v - 128;

    int r = (298 * c + 409 * e + 128) >> 8;
    int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int b = (298 * c + 516 * d + 128) >> 8;

    r = min(255, max(0, r));
    g = min(255, max(0, g));
    b = min(255, max(0, b));

    const size_t idx = static_cast<size_t>(py) * dst_pitch + px * 3;
    dst[idx + 0] = static_cast<unsigned char>(b);
    dst[idx + 1] = static_cast<unsigned char>(g);
    dst[idx + 2] = static_cast<unsigned char>(r);
}

void launchNv12ToBgrKernel(
    const unsigned char* src_y, int src_pitch_y,
    const unsigned char* src_uv, int src_pitch_uv,
    unsigned char* dst, int dst_pitch,
    int width, int height,
    cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    nv12ToBgrKernel<<<grid_size, block_size, 0, stream>>>(
        src_y, src_pitch_y, src_uv, src_pitch_uv,
        dst, dst_pitch, width, height);
}

// ============================================================
// Wrapper functions (C++ callable)
// ============================================================

void launchResizeNormalizeKernel(
    const unsigned char* src,
    int src_w, int src_h, size_t src_pitch,
    float* dst,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size((dst_w + 15) / 16, (dst_h + 15) / 16);
    resizeNormalizeKernel<<<grid_size, block_size, 0, stream>>>(
        src, src_w, src_h, src_pitch, dst, dst_w, dst_h,
        mean_r, mean_g, mean_b, std_r, std_g, std_b);
}

void launchLetterboxResizeNormalizeKernel(
    const unsigned char* src,
    int src_w, int src_h, size_t src_pitch,
    float* dst,
    int dst_w, int dst_h,
    int letter_w, int letter_h,
    int pad_x, int pad_y,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size((dst_w + 15) / 16, (dst_h + 15) / 16);
    letterboxResizeNormalizeKernel<<<grid_size, block_size, 0, stream>>>(
        src, src_w, src_h, src_pitch, dst, dst_w, dst_h,
        letter_w, letter_h, pad_x, pad_y,
        mean_r, mean_g, mean_b, std_r, std_g, std_b);
}

void launchDrawRectKernel(
    unsigned char* image, int width, int height, int pitch,
    int x, int y, int w, int h, int thickness,
    unsigned char b, unsigned char g, unsigned char r,
    cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    drawRectKernel<<<grid_size, block_size, 0, stream>>>(
        image, width, height, pitch,
        x, y, w, h, thickness, b, g, r);
}
