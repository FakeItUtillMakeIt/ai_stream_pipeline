// src/hal/npp/npp_image_accelerator.cpp
// NVIDIA NPP 图像加速实现——封装 CUDA kernel 到 HAL 接口
#include "npp_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <algorithm>
#include <cmath>

// 声明 CUDA kernel（定义在 cuda_kernels_wrapper.cu 和 gpu_osd_draw.cu 中）
extern void launchResizeNormalizeKernel(
    const unsigned char* src,
    int src_w, int src_h, size_t src_pitch,
    float* dst,
    int dst_w, int dst_h,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    cudaStream_t stream);

extern void launchLetterboxResizeNormalizeKernel(
    const unsigned char* src,
    int src_w, int src_h, size_t src_pitch,
    float* dst,
    int dst_w, int dst_h,
    int letter_w, int letter_h,
    int pad_x, int pad_y,
    float mean_r, float mean_g, float mean_b,
    float std_r, float std_g, float std_b,
    cudaStream_t stream);

extern void launchDrawRectKernel(
    unsigned char* image, int width, int height, int pitch,
    int x, int y, int w, int h, int thickness,
    unsigned char b, unsigned char g, unsigned char r,
    cudaStream_t stream);

namespace ai_stream {
namespace hal {

NppImageAccelerator::NppImageAccelerator() {
    initCuda();
}

NppImageAccelerator::~NppImageAccelerator() {
    // CUDA 资源由节点管理，这里不释放
}

bool NppImageAccelerator::initCuda() {
    if (cuda_initialized_) return true;

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        LOG_WARN("[NppImageAccelerator] No GPU device found");
        return false;
    }

    device_id_ = 0;
    cudaSetDevice(device_id_);
    cuda_initialized_ = true;
    LOG_INFO_FMT("[NppImageAccelerator] Initialized with GPU device {}", device_id_);
    return true;
}

void NppImageAccelerator::calcLetterbox(const ResizeNormalizeParams& params, LetterboxResult& letter) {
    float scale = std::min(
        static_cast<float>(params.dst_width) / params.src_width,
        static_cast<float>(params.dst_height) / params.src_height);

    letter.letter_w = std::max(1, static_cast<int>(std::round(params.src_width * scale)));
    letter.letter_h = std::max(1, static_cast<int>(std::round(params.src_height * scale)));
    letter.pad_x = (params.dst_width - letter.letter_w) / 2;
    letter.pad_y = (params.dst_height - letter.letter_h) / 2;
    letter.scale = scale;
}

bool NppImageAccelerator::resizeNormalize(
    const uint8_t* src,
    const ResizeNormalizeParams& params,
    float* dst,
    LetterboxResult* letter) {

    if (!src || !dst || params.src_width <= 0 || params.src_height <= 0) {
        return false;
    }

    if (!cuda_initialized_) {
        LOG_ERROR("[NppImageAccelerator] CUDA not initialized");
        return false;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(params.stream);
    float mean_r = params.mean.size() > 0 ? params.mean[0] : 0.0f;
    float mean_g = params.mean.size() > 1 ? params.mean[1] : 0.0f;
    float mean_b = params.mean.size() > 2 ? params.mean[2] : 0.0f;
    float std_r = params.std.size() > 0 ? params.std[0] : 1.0f;
    float std_g = params.std.size() > 1 ? params.std[1] : 1.0f;
    float std_b = params.std.size() > 2 ? params.std[2] : 1.0f;

    if (params.keep_aspect_ratio) {
        LetterboxResult lb;
        calcLetterbox(params, lb);

        launchLetterboxResizeNormalizeKernel(
            src, params.src_width, params.src_height, params.src_pitch,
            dst, params.dst_width, params.dst_height,
            lb.letter_w, lb.letter_h, lb.pad_x, lb.pad_y,
            mean_r, mean_g, mean_b, std_r, std_g, std_b,
            stream);

        if (letter) *letter = lb;
    } else {
        launchResizeNormalizeKernel(
            src, params.src_width, params.src_height, params.src_pitch,
            dst, params.dst_width, params.dst_height,
            mean_r, mean_g, mean_b, std_r, std_g, std_b,
            stream);
    }

    return true;
}

bool NppImageAccelerator::drawBoxes(
    const std::vector<BBox>& boxes,
    const DrawParams& draw) {

    if (!draw.bgr || draw.width <= 0 || draw.height <= 0) {
        return false;
    }

    if (!cuda_initialized_) {
        LOG_ERROR("[NppImageAccelerator] CUDA not initialized");
        return false;
    }

    cudaStream_t stream = static_cast<cudaStream_t>(draw.stream);

    for (const auto& box : boxes) {
        // 类别过滤
        if (!draw.class_filter.empty()) {
            if (std::find(draw.class_filter.begin(), draw.class_filter.end(), box.class_id) == draw.class_filter.end()) {
                continue;
            }
        }

        int x = static_cast<int>(box.x);
        int y = static_cast<int>(box.y);
        int w = static_cast<int>(box.w);
        int h = static_cast<int>(box.h);

        launchDrawRectKernel(
            draw.bgr, draw.width, draw.height, draw.pitch,
            x, y, w, h, draw.font_thickness,
            static_cast<unsigned char>(draw.box_color_b),
            static_cast<unsigned char>(draw.box_color_g),
            static_cast<unsigned char>(draw.box_color_r),
            stream);
    }

    return true;
}

bool NppImageAccelerator::nms(std::vector<BBox>& boxes, float iou_threshold) {
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

            float x1 = std::max(boxes[i].x, boxes[j].x);
            float y1 = std::max(boxes[i].y, boxes[j].y);
            float x2 = std::min(boxes[i].x + boxes[i].w, boxes[j].x + boxes[j].w);
            float y2 = std::min(boxes[i].y + boxes[i].h, boxes[j].y + boxes[j].h);

            float inter_w = std::max(0.0f, x2 - x1);
            float inter_h = std::max(0.0f, y2 - y1);
            float inter_area = inter_w * inter_h;

            float area_i = boxes[i].w * boxes[i].h;
            float area_j = boxes[j].w * boxes[j].h;
            float union_area = area_i + area_j - inter_area;

            if (union_area > 0 && inter_area / union_area > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    boxes = std::move(result);
    return true;
}

bool NppImageAccelerator::isAvailable() const {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
}

// 注册到工厂
REGISTER_IMAGE_ACCELERATOR(ImageAcceleratorBackend::NPP, NppImageAccelerator)

} // namespace hal
} // namespace ai_stream
