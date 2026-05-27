// src/utils/cuda_utils.h
#pragma once

#include <cuda_runtime.h>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

void launchNV12ToBGR(
    const uint8_t* y_plane, const uint8_t* uv_plane,
    int width, int height,
    size_t y_pitch, size_t uv_pitch,
    uint8_t* bgr, size_t bgr_pitch,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif