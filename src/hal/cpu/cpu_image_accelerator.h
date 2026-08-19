// src/hal/cpu/cpu_image_accelerator.h
// CPU 图像加速实现——基于 OpenCV，无 GPU 依赖
#pragma once

#include "ai_stream/hal/i_image_accelerator.h"
#include <opencv2/core/mat.hpp>

namespace ai_stream {
namespace hal {

class CpuImageAccelerator : public IImageAccelerator {
public:
    CpuImageAccelerator() = default;
    ~CpuImageAccelerator() override = default;

    bool resizeNormalize(
        const uint8_t* src, int src_width, int src_height,
        float* dst, int dst_width, int dst_height,
        const std::vector<float>& mean,
        const std::vector<float>& std) override;

    bool drawBoxes(
        uint8_t* bgr, int width, int height, int pitch,
        const std::vector<BBox>& boxes) override;

    bool nms(std::vector<BBox>& boxes, float iou_threshold) override;

    std::string getName() const override { return "CPU (OpenCV)"; }
    bool isAvailable() const override { return true; }
};

} // namespace hal
} // namespace ai_stream
