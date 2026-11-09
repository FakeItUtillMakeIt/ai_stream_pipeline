// src/hal/image_accel/ascend/dvpp_image_accelerator.h
// DVPP 图像加速器——华为 Ascend 数字视觉预处理引擎
//
// resize 走 DVPP VPC 硬件（BGR888 in/out）；normalize（/255、mean/std、
// BGR→RGB、NCHW）与 NMS 仍在 CPU，DVPP 没有 NMS 算子。
// drawBoxes 用 OpenCV 回退，避免 OSD 节点选中 DVPP 后无法绘制。
#pragma once

#include "ai_stream/hal/i_image_accelerator.h"

#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>

#include <string>

namespace ai_stream {
namespace hal {

class DvppImageAccelerator : public IImageAccelerator {
public:
    DvppImageAccelerator();
    ~DvppImageAccelerator() override;

    bool resizeNormalize(const uint8_t* src,
                         const ResizeNormalizeParams& params,
                         float* dst,
                         LetterboxResult* letter = nullptr) override;

    bool drawBoxes(const std::vector<BBox>& boxes,
                   const DrawParams& draw) override;

    bool cvtColorNv12ToBgr(
        const void* src_y, int src_pitch_y,
        const void* src_uv, int src_pitch_uv,
        int width, int height,
        uint8_t* dst_bgr, int dst_pitch,
        void* stream = nullptr) override;

    bool nms(std::vector<BBox>& boxes, float iou_threshold) override;

    std::string getName() const override { return "DVPP VPC (Ascend)"; }
    bool isAvailable() const override;

    void setChannelCount(int count) { channel_count_ = count; }

private:
    bool initDvpp();
    void cleanup();

    bool initialized_ = false;
    int channel_count_ = 4;

    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    acldvppChannelDesc* channel_ = nullptr;
    acldvppResizeConfig* resize_config_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
