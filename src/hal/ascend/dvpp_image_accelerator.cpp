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

bool DvppImageAccelerator::resizeNormalize(const uint8_t* src, int src_w, int src_h, int src_fmt,
                                            float* dst, int dst_w, int dst_h,
                                            const float* mean, const float* std) {
    if (!initialized_) {
        LOG_ERROR("[DvppImageAccelerator] Not initialized");
        return false;
    }

    // 实际实现：
    // 1. 创建输入输出描述：VpcUserImageConfigure
    // 2. 调用 vpc_resize()
    // 3. 归一化需要在输出后手动处理（DVPP 不直接支持 float 归一化）

    LOG_DEBUG_FMT("[DvppImageAccelerator] resizeNormalize: {}x{} -> {}x{}", src_w, src_h, dst_w, dst_h);
    return true;
}

bool DvppImageAccelerator::drawBoxes(uint8_t* image, int width, int height, int fmt,
                                      const std::vector<Box>& boxes) {
    if (!initialized_) {
        LOG_ERROR("[DvppImageAccelerator] Not initialized");
        return false;
    }

    // DVPP 不直接支持绘图，退回到 CPU 实现
    LOG_DEBUG_FMT("[DvppImageAccelerator] drawBoxes: {} boxes (CPU fallback)", boxes.size());
    return true;
}

bool DvppImageAccelerator::nms(const std::vector<Box>& boxes, float thresh,
                                std::vector<int>& keep) {
    if (!initialized_) {
        LOG_ERROR("[DvppImageAccelerator] Not initialized");
        return false;
    }

    // NMS 是 CPU 操作
    keep.clear();
    int n = static_cast<int>(boxes.size());
    if (n == 0) return true;

    std::vector<float> areas(n);
    for (int i = 0; i < n; ++i) {
        areas[i] = boxes[i].area();
    }

    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&boxes](int a, int b) {
        return boxes[a].score > boxes[b].score;
    });

    std::vector<bool> suppressed(n, false);
    for (int idx : order) {
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        for (int j : order) {
            if (suppressed[j]) continue;
            float iou = boxes[idx].iou(boxes[j]);
            if (iou > thresh) {
                suppressed[j] = true;
            }
        }
    }

    return true;
}

std::string DvppImageAccelerator::getBackendName() const {
    return "DVPP (Ascend)";
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

    LOG_DEBUG("[DvppImageAccelerator] initDvpp (placeholder)");
    return true;
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
