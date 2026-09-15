// src/hal/image_accel/horizon/horizon_image_accelerator.cpp
// Horizon BPU 图像加速器实现——使用 OpenCV CPU 路径
#include "horizon_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <cstring>

namespace ai_stream {
namespace hal {

bool HorizonImageAccelerator::resizeNormalize(
    const uint8_t* src,
    const ResizeNormalizeParams& params,
    float* dst,
    LetterboxResult* letter) {

    if (!src || !dst) return false;

    cv::Mat src_mat(params.src_height, params.src_width, CV_8UC3,
                     const_cast<uint8_t*>(src));

    int dst_w = params.dst_width;
    int dst_h = params.dst_height;

    if (params.keep_aspect_ratio) {
        // Letterbox resize
        float scale = std::min(static_cast<float>(dst_w) / params.src_width,
                                static_cast<float>(dst_h) / params.src_height);
        int new_w = static_cast<int>(params.src_width * scale);
        int new_h = static_cast<int>(params.src_height * scale);
        int pad_x = (dst_w - new_w) / 2;
        int pad_y = (dst_h - new_h) / 2;

        cv::Mat resized;
        cv::resize(src_mat, resized, cv::Size(new_w, new_h));

        cv::Mat padded(dst_h, dst_w, CV_8UC3, cv::Scalar(114, 114, 114));
        resized.copyTo(padded(cv::Rect(pad_x, pad_y, new_w, new_h)));

        if (letter) {
            letter->letter_w = new_w;
            letter->letter_h = new_h;
            letter->pad_x = pad_x;
            letter->pad_y = pad_y;
            letter->scale = scale;
        }

        src_mat = padded;
    } else {
        cv::resize(src_mat, src_mat, cv::Size(dst_w, dst_h));
    }

    // Convert to float NCHW with normalization
    src_mat.convertTo(src_mat, CV_32FC3, 1.0 / 255.0);

    // Split channels and apply mean/std
    std::vector<cv::Mat> channels(3);
    cv::split(src_mat, channels);

    size_t plane_size = dst_w * dst_h;
    for (int c = 0; c < 3; ++c) {
        float mean_val = (c < static_cast<int>(params.mean.size())) ? params.mean[c] : 0.0f;
        float std_val = (c < static_cast<int>(params.std.size())) ? params.std[c] : 1.0f;

        const float* ch_data = reinterpret_cast<const float*>(channels[c].data);
        float* out_ch = dst + c * plane_size;

        for (size_t i = 0; i < plane_size; ++i) {
            out_ch[i] = (ch_data[i] - mean_val) / (std_val + 1e-6f);
        }
    }

    return true;
}

bool HorizonImageAccelerator::drawBoxes(
    const std::vector<BBox>& boxes,
    const DrawParams& draw) {

    if (!draw.bgr || draw.width <= 0 || draw.height <= 0) return false;

    cv::Mat img(draw.height, draw.width, CV_8UC3, draw.bgr,
                draw.pitch > 0 ? draw.pitch : draw.width * 3);

    for (const auto& box : boxes) {
        int x1 = static_cast<int>(box.x);
        int y1 = static_cast<int>(box.y);
        int x2 = static_cast<int>(box.x + box.w);
        int y2 = static_cast<int>(box.y + box.h);

        cv::Scalar color(draw.box_color_b, draw.box_color_g, draw.box_color_r);
        cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, draw.font_thickness);

        if (draw.show_confidence || draw.draw_labels) {
            std::string label;
            if (draw.draw_labels && !box.class_name.empty()) {
                label = box.class_name;
            }
            if (draw.show_confidence) {
                if (!label.empty()) label += " ";
                char buf[32];
                snprintf(buf, sizeof(buf), "%.2f", box.confidence);
                label += buf;
            }

            if (!label.empty()) {
                int baseline = 0;
                cv::Size tsz = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX,
                                                0.5, draw.font_thickness, &baseline);
                cv::rectangle(img, cv::Point(x1, y1 - tsz.height - 4),
                              cv::Point(x1 + tsz.width, y1), color, -1);
                cv::putText(img, label, cv::Point(x1, y1 - 2),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255), draw.font_thickness);
            }
        }
    }

    return true;
}

bool HorizonImageAccelerator::nms(std::vector<BBox>& boxes, float iou_threshold) {
    if (boxes.empty()) return true;

    // Sort by confidence descending
    std::sort(boxes.begin(), boxes.end(),
              [](const BBox& a, const BBox& b) { return a.confidence > b.confidence; });

    std::vector<bool> suppressed(boxes.size(), false);

    for (size_t i = 0; i < boxes.size(); ++i) {
        if (suppressed[i]) continue;

        for (size_t j = i + 1; j < boxes.size(); ++j) {
            if (suppressed[j]) continue;
            if (boxes[i].class_id != boxes[j].class_id) continue;

            float ix1 = std::max(boxes[i].x, boxes[j].x);
            float iy1 = std::max(boxes[i].y, boxes[j].y);
            float ix2 = std::min(boxes[i].x + boxes[i].w, boxes[j].x + boxes[j].w);
            float iy2 = std::min(boxes[i].y + boxes[i].h, boxes[j].y + boxes[j].h);

            float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
            float area_i = boxes[i].w * boxes[i].h;
            float area_j = boxes[j].w * boxes[j].h;
            float iou = inter / (area_i + area_j - inter + 1e-6f);

            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }

    // Remove suppressed boxes
    std::vector<BBox> result;
    for (size_t i = 0; i < boxes.size(); ++i) {
        if (!suppressed[i]) {
            result.push_back(boxes[i]);
        }
    }
    boxes = std::move(result);

    return true;
}

bool HorizonImageAccelerator::isAvailable() const {
    return true;  // CPU fallback always available
}

REGISTER_IMAGE_ACCELERATOR(ImageAcceleratorBackend::HORIZON, HorizonImageAccelerator)

} // namespace hal
} // namespace ai_stream
