// src/utils/cuda_utils.cu
#include "hw_cuda_decode.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// 注意：这里用 __device__ 函数替代 std::min/max，避免 C++ 宏冲突
__device__ inline int cuda_min(int a, int b) { return a < b ? a : b; }
__device__ inline int cuda_max(int a, int b) { return a > b ? a : b; }

__global__ void nv12ToBGRKernel(
    const uint8_t* __restrict__ y_plane,
    
    const uint8_t* __restrict__ uv_plane,
    int width, int height,
    size_t y_pitch, size_t uv_pitch,
    uint8_t* __restrict__ bgr, size_t bgr_pitch)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    uint8_t y_val = y_plane[y * y_pitch + x];
    
    int uv_x = x / 2;
    int uv_y = y / 2;
    uint8_t u_val = uv_plane[uv_y * uv_pitch + uv_x * 2];
    uint8_t v_val = uv_plane[uv_y * uv_pitch + uv_x * 2 + 1];

    int c = y_val - 16;
    int d = u_val - 128;
    int e = v_val - 128;
    
    int r = (298 * c + 409 * e + 128) >> 8;
    int g = (298 * c - 100 * d - 208 * e + 128) >> 8;
    int b = (298 * c + 516 * d + 128) >> 8;
    
    r = cuda_max(0, cuda_min(255, r));
    g = cuda_max(0, cuda_min(255, g));
    b = cuda_max(0, cuda_min(255, b));

    uint8_t* bgr_row = bgr + y * bgr_pitch;
    bgr_row[x * 3 + 0] = b;
    bgr_row[x * 3 + 1] = g;
    bgr_row[x * 3 + 2] = r;
}

// C++ 可调用的包装函数
void launchNV12ToBGR(
    const uint8_t* y_plane, const uint8_t* uv_plane,
    int width, int height,
    size_t y_pitch, size_t uv_pitch,
    uint8_t* bgr, size_t bgr_pitch,
    cudaStream_t stream)
{
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);
    
    nv12ToBGRKernel<<<grid, block, 0, stream>>>(
        y_plane, uv_plane, width, height,
        y_pitch, uv_pitch, bgr, bgr_pitch
    );
}