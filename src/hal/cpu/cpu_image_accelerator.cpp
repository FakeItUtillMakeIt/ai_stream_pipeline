// src/hal/cpu/cpu_image_accelerator.cpp
// CPU 图像加速实现——基于 OpenCV，无 GPU 依赖
#include "cpu_image_accelerator.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>

namespace ai_stream {
namespace hal {

bool CpuImageAccelerator::resizeNormalize(
    const uint8_t* src, int src_width, int src_height,
    float* dst, int dst_width, int dst_height,
    const std::vector<float>& mean,
    const std::vector<float>& std) {

    if (!src || !dst || src_width <= 0 || src_height <= 0) {
        return false;
    }

    // 构造源 Mat（RGB/BGR uint8）
    cv::Mat src_mat(src_height, src_width, CV_8UC3, const_cast<uint8_t*>(src));

    // Resize
    cv::Mat resized;
    cv::resize(src_mat, resized, cv::Size(dst_width, dst_height), 0, 0, cv::INTER_LINEAR);

    // 转 float 并归一化到 [0, 1]
    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32FC3, 1.0 / 255.0);

    // 转换到 NCHW 布局
    const int plane_size = dst_width * dst_height;
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
        channels[c] = cv::Mat(dst_height, dst_width, CV_32FC1,
                              dst + c * plane_size);
    }
    cv::split(float_mat, channels);

    // 应用均值和标准差
    float m[3] = {mean.size() > 0 ? mean[0] : 0.0f,
                  mean.size() > 1 ? mean[1] : 0.0f,
                  mean.size() > 2 ? mean[2] : 0.0f};
    float s[3] = {std.size() > 0 ? std[0] : 1.0f,
                  std.size() > 1 ? std[1] : 1.0f,
                  std.size() > 2 ? std[2] : 1.0f};

    for (int c = 0; c < 3; ++c) {
        cv::Mat plane(dst_height, dst_width, CV_32FC1, dst + c * plane_size);
        plane = (plane - m[c]) / s[c];
    }

    return true;
}

bool CpuImageAccelerator::drawBoxes(
    uint8_t* bgr, int width, int height, int pitch,
    const std::vector<BBox>& boxes) {

    if (!bgr || width <= 0 || height <= 0) {
        return false;
    }

    cv::Mat img(height, width, CV_8UC3, bgr, pitch);

    for (const auto& box : boxes) {
        cv::Rect rect(static_cast<int>(box.x), static_cast<int>(box.y),
                      static_cast<int>(box.w), static_cast<int>(box.h));
        cv::rectangle(img, rect, cv::Scalar(0, 255, 0), 2);

        if (!box.class_name.empty()) {
            std::string label = box.class_name + " " +
                std::to_string(static_cast<int>(box.confidence * 100)) + "%";
            cv::putText(img, label,
                        cv::Point(static_cast<int>(box.x), static_cast<int>(box.y) - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }
    }

    return true;
}

bool CpuImageAccelerator::nms(std::vector<BBox>& boxes, float iou_threshold) {
    if (boxes.empty()) return true;

    // 按置信度排序
    std::sort(boxes.begin(), boxes.end(),
              [](const BBox& a, const BBox& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(boxes.size(), false);
    std::vector<BBox> result;

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;
        result.push_back(boxes[i]);

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;

            // 计算 IoU
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

} // namespace hal
} // namespace ai_stream
