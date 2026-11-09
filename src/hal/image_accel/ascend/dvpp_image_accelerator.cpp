// src/hal/image_accel/ascend/dvpp_image_accelerator.cpp
#include "dvpp_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "ascend/ascend_acl.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace ai_stream {
namespace hal {
namespace {

int alignUp(int v, int a) {
    return (v + a - 1) / a * a;
}

float clampChannelValue(float v) {
    return std::max(0.0f, std::min(1.0f, v));
}

} // namespace

DvppImageAccelerator::DvppImageAccelerator() {
    initialized_ = initDvpp();
}

DvppImageAccelerator::~DvppImageAccelerator() {
    cleanup();
}

bool DvppImageAccelerator::initDvpp() {
    if (!ensureAscendAclInitialized()) {
        LOG_WARN("[DvppImageAccelerator] aclInit failed, falling back to CPU accelerator");
        return false;
    }
    if (aclrtSetDevice(0) != ACL_SUCCESS ||
        aclrtCreateContext(&context_, 0) != ACL_SUCCESS ||
        aclrtSetCurrentContext(context_) != ACL_SUCCESS ||
        aclrtCreateStream(&stream_) != ACL_SUCCESS) {
        LOG_WARN("[DvppImageAccelerator] create context/stream failed");
        cleanup();
        return false;
    }

    channel_ = acldvppCreateChannelDesc();
    if (!channel_ || acldvppCreateChannel(channel_) != ACL_SUCCESS) {
        LOG_WARN("[DvppImageAccelerator] create DVPP channel failed");
        cleanup();
        return false;
    }
    resize_config_ = acldvppCreateResizeConfig();
    if (!resize_config_) {
        LOG_WARN("[DvppImageAccelerator] create resize config failed");
        cleanup();
        return false;
    }
    // 默认最近邻；与 CPU 的 INTER_LINEAR 有细微差异，但检测 task 足够
    acldvppSetResizeConfigInterpolation(resize_config_, 0);
    LOG_INFO("[DvppImageAccelerator] DVPP VPC channel ready");
    return true;
}

void DvppImageAccelerator::cleanup() {
    if (resize_config_) {
        acldvppDestroyResizeConfig(resize_config_);
        resize_config_ = nullptr;
    }
    if (channel_) {
        acldvppDestroyChannel(channel_);
        acldvppDestroyChannelDesc(channel_);
        channel_ = nullptr;
    }
    if (stream_) {
        aclrtDestroyStream(stream_);
        stream_ = nullptr;
    }
    if (context_) {
        aclrtDestroyContext(context_);
        context_ = nullptr;
    }
    initialized_ = false;
}

bool DvppImageAccelerator::isAvailable() const {
    return initialized_ && ::access("/dev/davinci0", R_OK | W_OK) == 0;
}

bool DvppImageAccelerator::resizeNormalize(
    const uint8_t* src,
    const ResizeNormalizeParams& params,
    float* dst,
    LetterboxResult* letter) {
    if (!initialized_ || !src || !dst ||
        params.src_width <= 0 || params.src_height <= 0 ||
        params.dst_width <= 0 || params.dst_height <= 0) {
        return false;
    }
    // DVPP 的 context 是线程相关的；resize_normalize 可能在 worker 线程调用
    if (aclrtSetCurrentContext(context_) != ACL_SUCCESS) {
        LOG_ERROR("[DvppImageAccelerator] set context failed");
        return false;
    }

    const int src_w = params.src_width;
    const int src_h = params.src_height;
    const int dst_w = params.dst_width;
    const int dst_h = params.dst_height;

    int letter_w = dst_w, letter_h = dst_h, pad_x = 0, pad_y = 0;
    float scale = 1.0f;
    if (params.keep_aspect_ratio) {
        scale = std::min(static_cast<float>(dst_w) / src_w,
                         static_cast<float>(dst_h) / src_h);
        letter_w = std::max(1, static_cast<int>(std::round(src_w * scale)));
        letter_h = std::max(1, static_cast<int>(std::round(src_h * scale)));
        pad_x = (dst_w - letter_w) / 2;
        pad_y = (dst_h - letter_h) / 2;
    }
    const int resize_w = params.keep_aspect_ratio ? letter_w : dst_w;
    const int resize_h = params.keep_aspect_ratio ? letter_h : dst_h;

    const int src_pitch = params.src_pitch > 0
                              ? static_cast<int>(params.src_pitch)
                              : src_w * 3;
    const int in_stride = alignUp(std::max(src_pitch, src_w * 3), 16);
    const int out_stride = alignUp(resize_w * 3, 16);
    const size_t in_size = static_cast<size_t>(in_stride) * src_h;
    const size_t out_size = static_cast<size_t>(out_stride) * resize_h;

    void* in_dev = nullptr;
    void* out_dev = nullptr;
    acldvppPicDesc* in_desc = nullptr;
    acldvppPicDesc* out_desc = nullptr;

    if (acldvppMalloc(&in_dev, in_size) != ACL_SUCCESS || !in_dev ||
        acldvppMalloc(&out_dev, out_size) != ACL_SUCCESS || !out_dev) {
        if (in_dev) acldvppFree(in_dev);
        if (out_dev) acldvppFree(out_dev);
        LOG_ERROR("[DvppImageAccelerator] acldvppMalloc failed");
        return false;
    }

    bool ok = true;
    if (src_pitch == in_stride) {
        ok = aclrtMemcpy(in_dev, in_size, src, static_cast<size_t>(src_pitch) * src_h,
                         ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    } else {
        std::vector<uint8_t> staging(in_size, 0);
        for (int y = 0; y < src_h; ++y) {
            std::memcpy(staging.data() + static_cast<size_t>(y) * in_stride,
                        src + static_cast<size_t>(y) * src_pitch, src_w * 3);
        }
        ok = aclrtMemcpy(in_dev, in_size, staging.data(), in_size,
                         ACL_MEMCPY_HOST_TO_DEVICE) == ACL_SUCCESS;
    }
    if (!ok) {
        acldvppFree(in_dev);
        acldvppFree(out_dev);
        LOG_ERROR("[DvppImageAccelerator] H2D failed");
        return false;
    }

    in_desc = acldvppCreatePicDesc();
    out_desc = acldvppCreatePicDesc();
    if (!in_desc || !out_desc) {
        if (in_desc) acldvppDestroyPicDesc(in_desc);
        if (out_desc) acldvppDestroyPicDesc(out_desc);
        acldvppFree(in_dev);
        acldvppFree(out_dev);
        return false;
    }

    acldvppSetPicDescData(in_desc, in_dev);
    acldvppSetPicDescSize(in_desc, static_cast<uint32_t>(in_size));
    acldvppSetPicDescFormat(in_desc, PIXEL_FORMAT_BGR_888);
    acldvppSetPicDescWidth(in_desc, src_w);
    acldvppSetPicDescHeight(in_desc, src_h);
    acldvppSetPicDescWidthStride(in_desc, in_stride);
    acldvppSetPicDescHeightStride(in_desc, src_h);

    acldvppSetPicDescData(out_desc, out_dev);
    acldvppSetPicDescSize(out_desc, static_cast<uint32_t>(out_size));
    acldvppSetPicDescFormat(out_desc, PIXEL_FORMAT_BGR_888);
    acldvppSetPicDescWidth(out_desc, resize_w);
    acldvppSetPicDescHeight(out_desc, resize_h);
    acldvppSetPicDescWidthStride(out_desc, out_stride);
    acldvppSetPicDescHeightStride(out_desc, resize_h);

    aclError ret = acldvppVpcResizeAsync(channel_, in_desc, out_desc,
                                         resize_config_, stream_);
    if (ret == ACL_SUCCESS) {
        ret = aclrtSynchronizeStream(stream_);
    }
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[DvppImageAccelerator] VPC resize failed: {}", static_cast<int>(ret));
        acldvppDestroyPicDesc(in_desc);
        acldvppDestroyPicDesc(out_desc);
        acldvppFree(in_dev);
        acldvppFree(out_dev);
        return false;
    }

    std::vector<uint8_t> host_out(out_size);
    ok = aclrtMemcpy(host_out.data(), out_size, out_dev, out_size,
                     ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
    acldvppDestroyPicDesc(in_desc);
    acldvppDestroyPicDesc(out_desc);
    acldvppFree(in_dev);
    acldvppFree(out_dev);
    if (!ok) {
        LOG_ERROR("[DvppImageAccelerator] D2H failed");
        return false;
    }

    // CPU 归一化：BGR HWC uint8 → RGB NCHW float
    const float mean[3] = {
        params.mean.size() > 0 ? params.mean[0] : 0.0f,
        params.mean.size() > 1 ? params.mean[1] : 0.0f,
        params.mean.size() > 2 ? params.mean[2] : 0.0f};
    const float stdv[3] = {
        params.std.size() > 0 ? params.std[0] : 1.0f,
        params.std.size() > 1 ? params.std[1] : 1.0f,
        params.std.size() > 2 ? params.std[2] : 1.0f};
    const float gray = 114.0f / 255.0f;
    const size_t plane = static_cast<size_t>(dst_w) * dst_h;

    for (int y = 0; y < dst_h; ++y) {
        for (int x = 0; x < dst_w; ++x) {
            const size_t idx = static_cast<size_t>(y) * dst_w + x;
            const bool inside = !params.keep_aspect_ratio ||
                                (x >= pad_x && x < pad_x + resize_w &&
                                 y >= pad_y && y < pad_y + resize_h);
            float b, g, r;
            if (inside) {
                const int sx = params.keep_aspect_ratio ? x - pad_x : x;
                const int sy = params.keep_aspect_ratio ? y - pad_y : y;
                const uint8_t* p = host_out.data() +
                                   static_cast<size_t>(sy) * out_stride + sx * 3;
                b = p[0] / 255.0f;
                g = p[1] / 255.0f;
                r = p[2] / 255.0f;
            } else {
                b = g = r = gray;
            }
            dst[0 * plane + idx] = stdv[0] != 0.0f ? (r - mean[0]) / stdv[0] : (r - mean[0]);
            dst[1 * plane + idx] = stdv[1] != 0.0f ? (g - mean[1]) / stdv[1] : (g - mean[1]);
            dst[2 * plane + idx] = stdv[2] != 0.0f ? (b - mean[2]) / stdv[2] : (b - mean[2]);
        }
    }
    (void)clampChannelValue;

    if (letter && params.keep_aspect_ratio) {
        letter->letter_w = letter_w;
        letter->letter_h = letter_h;
        letter->pad_x = pad_x;
        letter->pad_y = pad_y;
        letter->scale = scale;
    }
    return true;
}

bool DvppImageAccelerator::drawBoxes(
    const std::vector<BBox>& boxes,
    const DrawParams& draw) {
    if (draw.is_gpu || !draw.bgr || draw.width <= 0 || draw.height <= 0) {
        return false;
    }
    // DVPP 没有“画到内存”的 2D 硬件，OSD 用 CPU OpenCV 回退
    cv::Mat img(draw.height, draw.width, CV_8UC3, draw.bgr, draw.pitch);
    for (const auto& box : boxes) {
        if (!draw.class_filter.empty() &&
            std::find(draw.class_filter.begin(), draw.class_filter.end(),
                      box.class_id) == draw.class_filter.end()) {
            continue;
        }
        cv::rectangle(img,
                      cv::Rect(static_cast<int>(box.x), static_cast<int>(box.y),
                               static_cast<int>(box.w), static_cast<int>(box.h)),
                      cv::Scalar(draw.box_color_b, draw.box_color_g, draw.box_color_r),
                      draw.font_thickness);
        if (draw.draw_labels && !box.class_name.empty()) {
            std::string label = box.class_name;
            if (draw.show_confidence) {
                label += " " + std::to_string(static_cast<int>(box.confidence * 100)) + "%";
            }
            cv::putText(img, label,
                        cv::Point(static_cast<int>(box.x), static_cast<int>(box.y) - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5,
                        cv::Scalar(draw.box_color_b, draw.box_color_g, draw.box_color_r), 1);
        }
    }
    return true;
}

bool DvppImageAccelerator::cvtColorNv12ToBgr(
    const void* src_y, int src_pitch_y,
    const void* src_uv, int src_pitch_uv,
    int width, int height,
    uint8_t* dst_bgr, int dst_pitch,
    void* /*stream*/) {
    if (!src_y || !src_uv || !dst_bgr || width <= 0 || height <= 0) {
        return false;
    }
    try {
        cv::Mat y_plane(height, width, CV_8UC1,
                        const_cast<void*>(src_y), src_pitch_y);
        cv::Mat uv_plane(height / 2, width / 2, CV_8UC2,
                         const_cast<void*>(src_uv), src_pitch_uv);
        cv::Mat dst(height, width, CV_8UC3, dst_bgr, dst_pitch);
        cv::cvtColorTwoPlane(y_plane, uv_plane, dst, cv::COLOR_YUV2BGR_NV12);
        return true;
    } catch (const cv::Exception& e) {
        LOG_ERROR_FMT("[DvppImageAccelerator] NV12->BGR failed: {}", e.what());
        return false;
    }
}

bool DvppImageAccelerator::nms(std::vector<BBox>& boxes, float iou_threshold) {    // DVPP 无 NMS 算子；保持与 CPU 后端同语义（跨类别抑制）
    if (boxes.empty()) return true;
    std::sort(boxes.begin(), boxes.end(),
              [](const BBox& a, const BBox& b) { return a.confidence > b.confidence; });
    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BBox> result;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);
        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            const float x1 = std::max(boxes[i].x, boxes[j].x);
            const float y1 = std::max(boxes[i].y, boxes[j].y);
            const float x2 = std::min(boxes[i].x + boxes[i].w, boxes[j].x + boxes[j].w);
            const float y2 = std::min(boxes[i].y + boxes[i].h, boxes[j].y + boxes[j].h);
            const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
            const float area_i = boxes[i].w * boxes[i].h;
            const float area_j = boxes[j].w * boxes[j].h;
            const float uni = area_i + area_j - inter;
            if (uni > 0 && inter / uni > iou_threshold) suppressed[j] = true;
        }
    }
    boxes = std::move(result);
    return true;
}

REGISTER_IMAGE_ACCELERATOR(ImageAcceleratorBackend::DVPP, DvppImageAccelerator)

} // namespace hal
} // namespace ai_stream
