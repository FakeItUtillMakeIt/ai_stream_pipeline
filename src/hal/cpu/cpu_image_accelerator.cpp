// src/hal/cpu/cpu_image_accelerator.cpp
// CPU 图像加速实现——基于 OpenCV，无 GPU 依赖
#include "cpu_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace ai_stream {
namespace hal {

bool CpuImageAccelerator::resizeNormalize(
    const uint8_t* src,
    const ResizeNormalizeParams& params,
    float* dst,
    LetterboxResult* letter) {

    if (!src || !dst || params.src_width <= 0 || params.src_height <= 0) {
        return false;
    }

    cv::Mat src_mat(params.src_height, params.src_width, CV_8UC3,
                    const_cast<uint8_t*>(src),
                    params.src_pitch > 0 ? params.src_pitch : cv::Mat::AUTO_STEP);

    int dst_w = params.dst_width;
    int dst_h = params.dst_height;
    int letter_w = dst_w, letter_h = dst_h, pad_x = 0, pad_y = 0;
    float scale = 1.0f;

    if (params.keep_aspect_ratio) {
        scale = std::min(
            static_cast<float>(params.dst_width) / params.src_width,
            static_cast<float>(params.dst_height) / params.src_height);
        letter_w = std::max(1, static_cast<int>(std::round(params.src_width * scale)));
        letter_h = std::max(1, static_cast<int>(std::round(params.src_height * scale)));
        pad_x = (dst_w - letter_w) / 2;
        pad_y = (dst_h - letter_h) / 2;
    }

    // Resize
    cv::Mat resized;
    cv::resize(src_mat, resized, cv::Size(letter_w, letter_h), 0, 0, cv::INTER_LINEAR);

    // 转 float 并归一化到 [0, 1]
    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32FC3, 1.0 / 255.0);

    // 创建目标 Mat（NCHW 布局）
    const int plane_size = dst_w * dst_h;
    cv::Mat dst_mat(dst_h, dst_w, CV_32FC3, dst);

    // 填充灰色背景（letterbox 时）
    if (params.keep_aspect_ratio) {
        dst_mat.setTo(cv::Scalar(114.0f / 255.0f, 114.0f / 255.0f, 114.0f / 255.0f));
        float_mat.copyTo(dst_mat(cv::Rect(pad_x, pad_y, letter_w, letter_h)));
    } else {
        float_mat.copyTo(dst_mat);
    }

    // BGR HWC -> RGB NCHW，并应用均值和标准差
    std::vector<cv::Mat> bgr_channels;
    cv::split(dst_mat, bgr_channels);
    float m[3] = {params.mean.size() > 0 ? params.mean[0] : 0.0f,
                  params.mean.size() > 1 ? params.mean[1] : 0.0f,
                  params.mean.size() > 2 ? params.mean[2] : 0.0f};
    float s[3] = {params.std.size() > 0 ? params.std[0] : 1.0f,
                  params.std.size() > 1 ? params.std[1] : 1.0f,
                  params.std.size() > 2 ? params.std[2] : 1.0f};

    for (int c = 0; c < 3; ++c) {
        cv::Mat output_plane(dst_h, dst_w, CV_32FC1, dst + c * plane_size);
        cv::subtract(bgr_channels[2 - c], m[c], output_plane);
        if (s[c] != 0.0f) {
            cv::divide(output_plane, s[c], output_plane);
        }
    }

    // 输出 letterbox 参数
    if (letter && params.keep_aspect_ratio) {
        letter->letter_w = letter_w;
        letter->letter_h = letter_h;
        letter->pad_x = pad_x;
        letter->pad_y = pad_y;
        letter->scale = scale;
    }

    return true;
}

bool CpuImageAccelerator::drawBoxes(
    const std::vector<BBox>& boxes,
    const DrawParams& draw) {

    if (!draw.bgr || draw.width <= 0 || draw.height <= 0) {
        return false;
    }

    cv::Mat img(draw.height, draw.width, CV_8UC3, draw.bgr, draw.pitch);

    for (const auto& box : boxes) {
        // 类别过滤
        if (!draw.class_filter.empty()) {
            if (std::find(draw.class_filter.begin(), draw.class_filter.end(), box.class_id) == draw.class_filter.end()) {
                continue;
            }
        }

        cv::Rect rect(static_cast<int>(box.x), static_cast<int>(box.y),
                      static_cast<int>(box.w), static_cast<int>(box.h));
        cv::Scalar color(draw.box_color_b, draw.box_color_g, draw.box_color_r);
        cv::rectangle(img, rect, color, draw.font_thickness);

        if (!box.class_name.empty()) {
            std::string label = box.class_name;
            if (draw.show_confidence) {
                label += " " + std::to_string(static_cast<int>(box.confidence * 100)) + "%";
            }
            cv::putText(img, label,
                        cv::Point(static_cast<int>(box.x), static_cast<int>(box.y) - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1);
        }
    }

    return true;
}

bool CpuImageAccelerator::nms(std::vector<BBox>& boxes, float iou_threshold) {
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

REGISTER_IMAGE_ACCELERATOR(ImageAcceleratorBackend::CPU, CpuImageAccelerator)

} // namespace hal
} // namespace ai_stream
