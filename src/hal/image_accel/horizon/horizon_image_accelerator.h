// src/hal/image_accel/horizon/horizon_image_accelerator.h
// Horizon BPU 图像加速器——地平线 RDK S100P
// 使用 OpenCV 实现 CPU 路径（BPU 暂无专用图像加速 API）
#pragma once

#include "ai_stream/hal/i_image_accelerator.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <mutex>

namespace ai_stream {
namespace hal {

class HorizonImageAccelerator : public IImageAccelerator {
public:
    HorizonImageAccelerator() = default;
    ~HorizonImageAccelerator() override = default;

    bool resizeNormalize(
        const uint8_t* src,
        const ResizeNormalizeParams& params,
        float* dst,
        LetterboxResult* letter = nullptr) override;

    bool drawBoxes(
        const std::vector<BBox>& boxes,
        const DrawParams& draw) override;

    bool nms(std::vector<BBox>& boxes, float iou_threshold) override;

    std::string getName() const override { return "Horizon BPU ImageAccel (CPU fallback)"; }
    bool isAvailable() const override;

private:
    std::mutex mutex_;
};

} // namespace hal
} // namespace ai_stream
