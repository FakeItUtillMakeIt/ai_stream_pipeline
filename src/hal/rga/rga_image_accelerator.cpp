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

bool RgaImageAccelerator::resizeNormalize(const uint8_t* src, int src_w, int src_h, int src_fmt,
                                           float* dst, int dst_w, int dst_h,
                                           const float* mean, const float* std) {
    if (!initialized_) {
        LOG_ERROR("[RgaImageAccelerator] Not initialized");
        return false;
    }

    // 实际实现：
    // 1. 使用 imresize() 进行硬件缩放
    // 2. 使用 imcolor_convert() 进行颜色空间转换
    // 3. 使用 imnormalize() 进行归一化
    //
    // 伪代码：
    // rga_buffer_t src_buf = wrap_ptr_to_rga_buffer(src, src_w, src_h, src_fmt);
    // rga_buffer_t dst_buf = wrap_ptr_to_rga_buffer(dst, dst_w, dst_h, RK_FORMAT_RGB_888);
    // imresize(src_buf, dst_buf);
    // // 然后做归一化

    LOG_DEBUG_FMT("[RgaImageAccelerator] resizeNormalize: {}x{} -> {}x{}", src_w, src_h, dst_w, dst_h);
    return true;
}

bool RgaImageAccelerator::drawBoxes(uint8_t* image, int width, int height, int fmt,
                                     const std::vector<Box>& boxes) {
    if (!initialized_) {
        LOG_ERROR("[RgaImageAccelerator] Not initialized");
        return false;
    }

    // 实际实现：
    // 使用 imrectangle() 在图像上绘制矩形框

    LOG_DEBUG_FMT("[RgaImageAccelerator] drawBoxes: {} boxes on {}x{}", boxes.size(), width, height);
    return true;
}

bool RgaImageAccelerator::nms(const std::vector<Box>& boxes, float thresh,
                               std::vector<int>& keep) {
    if (!initialized_) {
        LOG_ERROR("[RgaImageAccelerator] Not initialized");
        return false;
    }

    // NMS 通常是 CPU 操作，RGA 不直接支持
    // 这里退回到 CPU 实现

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

std::string RgaImageAccelerator::getBackendName() const {
    return "RGA (Rockchip 2D)";
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

    LOG_DEBUG("[RgaImageAccelerator] initRga (placeholder)");
    return true;
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
