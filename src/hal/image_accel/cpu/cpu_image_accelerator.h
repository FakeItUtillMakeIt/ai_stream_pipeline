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
        const uint8_t* src,
        const ResizeNormalizeParams& params,
        float* dst,
        LetterboxResult* letter = nullptr) override;

    bool drawBoxes(
        const std::vector<BBox>& boxes,
        const DrawParams& draw) override;

    bool cvtColorNv12ToBgr(
        const void* src_y, int src_pitch_y,
        const void* src_uv, int src_pitch_uv,
        int width, int height,
        uint8_t* dst_bgr, int dst_pitch,
        void* stream = nullptr) override;

    bool nms(std::vector<BBox>& boxes, float iou_threshold) override;

    std::string getName() const override { return "CPU (OpenCV)"; }
    bool isAvailable() const override { return true; }
};

} // namespace hal
} // namespace ai_stream
