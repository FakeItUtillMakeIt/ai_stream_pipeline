// src/hal/ascend/dvpp_image_accelerator.h
// DVPP 图像加速器——华为 Ascend 数字视觉预处理引擎
#pragma once

#include "ai_stream/hal/i_image_accelerator.h"
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief DVPP 图像加速器
 *
 * 使用华为 Ascend DVPP (Digital Video Pre-Processing) 引擎进行：
 * - 图像/视频编解码 (JPEG decode/encode, H.264/H.265 decode/encode)
 * - 图像缩放 (resize)
 * - 裁剪 (crop)
 * - 颜色空间转换
 *
 * 通过 ACL (Ascend Computing Language) API 调用。
 */
class DvppImageAccelerator : public IImageAccelerator {
public:
    DvppImageAccelerator();
    ~DvppImageAccelerator() override;

    bool resizeNormalize(const uint8_t* src, int src_w, int src_h, int src_fmt,
                         float* dst, int dst_w, int dst_h,
                         const float* mean, const float* std) override;

    bool drawBoxes(uint8_t* image, int width, int height, int fmt,
                   const std::vector<Box>& boxes) override;

    bool nms(const std::vector<Box>& boxes, float thresh,
             std::vector<int>& keep) override;

    std::string getBackendName() const override { return "DVPP (Ascend)"; }
    bool isAvailable() const override;

    /**
     * @brief 设置 DVPP 通道数
     */
    void setChannelCount(int count);

private:
    bool initDvpp();
    void cleanup();

    bool initialized_ = false;
    int channel_count_ = 4;
    void* dvpp_ctx_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
