// src/hal/npp/npp_image_accelerator.h
// NVIDIA NPP 图像加速实现——封装 CUDA kernel 到 HAL 接口
#pragma once

#include "ai_stream/hal/i_image_accelerator.h"
#include <cuda_runtime.h>

namespace ai_stream {
namespace hal {

/**
 * @brief NVIDIA NPP 图像加速器
 *
 * 封装 CUDA kernel 实现的 resize+normalize 和 OSD 绘制。
 * 与 ResizeNormalizeNode 的 GPU 路径和 OSDDrawNode GPU 绘制功能等价。
 */
class NppImageAccelerator : public IImageAccelerator {
public:
    NppImageAccelerator();
    ~NppImageAccelerator() override;

    bool resizeNormalize(
        const uint8_t* src,
        const ResizeNormalizeParams& params,
        float* dst,
        LetterboxResult* letter = nullptr) override;

    bool drawBoxes(
        const std::vector<BBox>& boxes,
        const DrawParams& draw) override;

    bool nms(std::vector<BBox>& boxes, float iou_threshold) override;

    std::string getName() const override { return "NPP (NVIDIA GPU)"; }
    bool isAvailable() const override;

private:
    bool initCuda();
    void calcLetterbox(const ResizeNormalizeParams& params, LetterboxResult& letter);

    int device_id_ = 0;
    bool cuda_initialized_ = false;
};

} // namespace hal
} // namespace ai_stream
