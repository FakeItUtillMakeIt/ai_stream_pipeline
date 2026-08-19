// src/hal/rga/rga_image_accelerator.h
// RGA 图像加速器——Rockchip RK3588 2D 图形加速引擎
#pragma once

#include "ai_stream/hal/i_image_accelerator.h"
#include <string>

// 前向声明 RGA API 类型
struct rga_ctx;

namespace ai_stream {
namespace hal {

/**
 * @brief RGA 图像加速器
 *
 * 使用 Rockchip RGA (Raster Graphic Acceleration) 引擎进行：
 * - 图像缩放 (resize)
 * - 颜色空间转换 (color conversion)
 * - 图像叠加 (overlay/blend)
 * - 旋转 (rotation)
 *
 * RK3588 包含 3 个 RGA 核心，支持并行处理。
 */
class RgaImageAccelerator : public IImageAccelerator {
public:
    RgaImageAccelerator();
    ~RgaImageAccelerator() override;

    bool resizeNormalize(const uint8_t* src, int src_w, int src_h, int src_fmt,
                         float* dst, int dst_w, int dst_h,
                         const float* mean, const float* std) override;

    bool drawBoxes(uint8_t* image, int width, int height, int fmt,
                   const std::vector<Box>& boxes) override;

    bool nms(const std::vector<Box>& boxes, float thresh,
             std::vector<int>& keep) override;

    std::string getBackendName() const override { return "RGA (Rockchip 2D)"; }
    bool isAvailable() const override;

    /**
     * @brief 设置 RGA 核心掩码
     * @param core_mask 核心掩码，如 0x07 表示使用 3 个核心
     */
    void setCoreMask(int core_mask);

private:
    bool initRga();
    void cleanup();

    bool initialized_ = false;
    int core_mask_ = 0;
    void* rga_ctx_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
