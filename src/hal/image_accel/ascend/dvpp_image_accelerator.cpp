// src/hal/ascend/dvpp_image_accelerator.cpp
// DVPP 图像加速器——华为 Ascend 数字视觉预处理引擎
#include "dvpp_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

// 实际使用时取消注释：
// #include <acl/acl.h>
// #include <acl/ops/acl_dvpp.h>

namespace ai_stream {
namespace hal {

DvppImageAccelerator::DvppImageAccelerator() {
    initialized_ = initDvpp();
    if (initialized_) {
        LOG_DEBUG("[DvppImageAccelerator] Initialized");
    }
}

DvppImageAccelerator::~DvppImageAccelerator() {
    cleanup();
    LOG_DEBUG("[DvppImageAccelerator] Destroyed");
}

bool DvppImageAccelerator::resizeNormalize(const uint8_t* src,
                                           const ResizeNormalizeParams& params,
                                           float* dst, LetterboxResult* letter) {
    (void)src; (void)params; (void)dst; (void)letter;
    LOG_ERROR("[DvppImageAccelerator] DVPP SDK implementation is not available");
    return false;
}

bool DvppImageAccelerator::drawBoxes(const std::vector<BBox>& boxes,
                                     const DrawParams& draw) {
    (void)boxes; (void)draw;
    LOG_ERROR("[DvppImageAccelerator] DVPP does not implement drawing");
    return false;
}

bool DvppImageAccelerator::nms(std::vector<BBox>& boxes, float iou_threshold) {
    (void)boxes; (void)iou_threshold;
    LOG_ERROR("[DvppImageAccelerator] DVPP SDK implementation is not available");
    return false;
}

bool DvppImageAccelerator::isAvailable() const {
#ifdef WITH_ASCEND
    // 检查 Ascend 设备是否存在
    // return aclrtGetDeviceCount() > 0;
    return initialized_;
#else
    return false;
#endif
}

void DvppImageAccelerator::setChannelCount(int count) {
    channel_count_ = count;
}

bool DvppImageAccelerator::initDvpp() {
    // 实际实现：
    // 1. aclInit()
    // 2. aclrtSetDevice(device_id)
    // 3. acldvppCreateChannelDesc()
    // 4. acldvppCreateChannel()

    LOG_WARN("[DvppImageAccelerator] DVPP backend is not implemented");
    return false;
}

void DvppImageAccelerator::cleanup() {
    // 实际实现：
    // acldvppDestroyChannel()
    // aclrtResetDevice()

    initialized_ = false;
}

// 注册 DVPP 后端到工厂
#ifdef WITH_ASCEND
REGISTER_IMAGE_ACCELERATOR(ImageAcceleratorBackend::DVPP, DvppImageAccelerator)
#endif

} // namespace hal
} // namespace ai_stream
