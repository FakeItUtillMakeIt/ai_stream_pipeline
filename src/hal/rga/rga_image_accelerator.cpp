// src/hal/rga/rga_image_accelerator.cpp
// RGA 图像加速器——Rockchip RK3588 2D 图形加速引擎
#include "rga_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

// 实际使用时取消注释：
// #include <rga/RgaApi.h>
// #include <rga/im2d.h>

namespace ai_stream {
namespace hal {

RgaImageAccelerator::RgaImageAccelerator() {
    initialized_ = initRga();
    if (initialized_) {
        LOG_DEBUG("[RgaImageAccelerator] Initialized");
    }
}

RgaImageAccelerator::~RgaImageAccelerator() {
    cleanup();
    LOG_DEBUG("[RgaImageAccelerator] Destroyed");
}

bool RgaImageAccelerator::resizeNormalize(const uint8_t* src,
                                          const ResizeNormalizeParams& params,
                                          float* dst, LetterboxResult* letter) {
    (void)src; (void)params; (void)dst; (void)letter;
    LOG_ERROR("[RgaImageAccelerator] RGA SDK implementation is not available");
    return false;
}

bool RgaImageAccelerator::drawBoxes(const std::vector<BBox>& boxes,
                                    const DrawParams& draw) {
    (void)boxes; (void)draw;
    LOG_ERROR("[RgaImageAccelerator] RGA SDK implementation is not available");
    return false;
}

bool RgaImageAccelerator::nms(std::vector<BBox>& boxes, float iou_threshold) {
    (void)boxes; (void)iou_threshold;
    LOG_ERROR("[RgaImageAccelerator] RGA SDK implementation is not available");
    return false;
}

bool RgaImageAccelerator::isAvailable() const {
#ifdef WITH_RKNN
    // 检查 RGA 设备是否存在
    // return access("/dev/rga", F_OK) == 0;
    return initialized_;
#else
    return false;
#endif
}

void RgaImageAccelerator::setCoreMask(int core_mask) {
    core_mask_ = core_mask;
}

bool RgaImageAccelerator::initRga() {
    // 实际实现：
    // 初始化 RGA 设备，创建上下文
    // rga_init(&rga_ctx_);

    LOG_WARN("[RgaImageAccelerator] RGA backend is not implemented");
    return false;
}

void RgaImageAccelerator::cleanup() {
    // 实际实现：
    // rga_deinit(rga_ctx_);

    initialized_ = false;
}

// 注册 RGA 后端到工厂
#ifdef WITH_RKNN
REGISTER_IMAGE_ACCELERATOR(ImageAcceleratorBackend::RGA, RgaImageAccelerator)
#endif

} // namespace hal
} // namespace ai_stream
