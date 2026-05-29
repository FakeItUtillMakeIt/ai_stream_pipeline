// src/nodes/draw/gpu_osd_draw.cu
// 【加速优化】GPU 加速 OSD 绘制 - CUDA 实现
#include "gpu_osd_draw.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/cuda_check.h"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <filesystem>

namespace ai_stream {
namespace nodes {

namespace {

// CUDA 内核：绘制矩形边框
__global__ void drawRectKernel(
    unsigned char* image, int width, int height, int pitch,
    int x, int y, int w, int h, int thickness,
    unsigned char b, unsigned char g, unsigned char r)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    // 检查像素是否在矩形边框上
    bool on_border = false;

    // 上边
    if (py >= y && py < y + thickness && px >= x && px < x + w) {
        on_border = true;
    }
    // 下边
    if (py >= y + h - thickness && py < y + h && px >= x && px < x + w) {
        on_border = true;
    }
    // 左边
    if (px >= x && px < x + thickness && py >= y && py < y + h) {
        on_border = true;
    }
    // 右边
    if (px >= x + w - thickness && px < x + w && py >= y && py < y + h) {
        on_border = true;
    }

    if (on_border) {
        int idx = py * pitch + px * 3;
        image[idx + 0] = b;
        image[idx + 1] = g;
        image[idx + 2] = r;
    }
}

// CUDA 内核：批量绘制矩形
__global__ void drawBoxesBatchKernel(
    unsigned char* image, int width, int height, int pitch,
    const float* boxes, const float* scores, const int* class_ids,
    int num_boxes, int thickness,
    unsigned char b, unsigned char g, unsigned char r)
{
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= width || py >= height) return;

    for (int i = 0; i < num_boxes; ++i) {
        int bx = static_cast<int>(boxes[i * 4 + 0]);
        int by = static_cast<int>(boxes[i * 4 + 1]);
        int bw = static_cast<int>(boxes[i * 4 + 2]);
        int bh = static_cast<int>(boxes[i * 4 + 3]);

        bool on_border = false;

        // 上边
        if (py >= by && py < by + thickness && px >= bx && px < bx + bw) {
            on_border = true;
        }
        // 下边
        if (py >= by + bh - thickness && py < by + bh && px >= bx && px < bx + bw) {
            on_border = true;
        }
        // 左边
        if (px >= bx && px < bx + thickness && py >= by && py < by + bh) {
            on_border = true;
        }
        // 右边
        if (px >= bx + bw - thickness && px < bx + bw && py >= by && py < by + bh) {
            on_border = true;
        }

        if (on_border) {
            int idx = py * pitch + px * 3;
            image[idx + 0] = b;
            image[idx + 1] = g;
            image[idx + 2] = r;
            break; // 一个像素只画一次
        }
    }
}

} // anonymous namespace

GpuOSDDrawNode::GpuOSDDrawNode() : IDrawNode("GpuOSDDraw") {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        device_id_ = 0;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[GpuOSDDraw] Node created with GPU device {}", device_id_);
    } else {
        LOG_WARN_FMT("[GpuOSDDraw] No GPU device found");
    }

    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

GpuOSDDrawNode::~GpuOSDDrawNode() {
    if (stream_) cudaStreamDestroy(stream_);
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);
    LOG_INFO_FMT("[GpuOSDDraw] Node destroyed");
}

void GpuOSDDrawNode::setBoxColor(int b, int g, int r) {
    box_color_ = cv::Scalar(b, g, r);
}

void GpuOSDDrawNode::setFontThickness(int thickness) {
    font_thickness_ = thickness;
}

void GpuOSDDrawNode::setShowConfidence(bool show) {
    show_confidence_ = show;
}

void GpuOSDDrawNode::setClassFilter(const std::vector<int>& class_ids) {
    class_filter_ = class_ids;
}

bool GpuOSDDrawNode::start() {
    if (!stream_) {
        CUDA_CHECK_BOOL(cudaStreamCreate(&stream_));
    }

    // 如果启用了快照，创建保存目录
    if (snapshot_enabled_) {
        if (!std::filesystem::exists(snapshot_dir_)) {
            try {
                std::filesystem::create_directories(snapshot_dir_);
                LOG_INFO_FMT("[GpuOSDDraw] Created snapshot directory: {}", snapshot_dir_);
            } catch (const std::exception& e) {
                LOG_ERROR_FMT("[GpuOSDDraw] Failed to create snapshot directory: {}", e.what());
                snapshot_enabled_ = false;
            }
        }
    }

    running_ = true;
    LOG_INFO_FMT("[GpuOSDDraw] Started (snapshot: {}, interval: {})",
                 snapshot_enabled_.load(), snapshot_interval_.load());
    return true;
}

void GpuOSDDrawNode::stop() {
    running_ = false;
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }
    LOG_INFO_FMT("[GpuOSDDraw] Stopped");
}

void GpuOSDDrawNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO_FMT("[GpuOSDDraw] Received stream end");
        stop();
        broadcast(packet);
        return;
    }
    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }

    auto infer_result = std::static_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer_result->source_frame) {
        LOG_WARN_FMT("[GpuOSDDraw] Inference result missing source frame");
        return;
    }

    auto source_frame = infer_result->source_frame;

    // 过滤检测框
    std::vector<core::InferenceResultPacket::BBox> filtered_dets;
    for (const auto& det : infer_result->detections) {
        if (!class_filter_.empty() &&
            std::find(class_filter_.begin(), class_filter_.end(), det.class_id) == class_filter_.end()) {
            continue;
        }
        filtered_dets.push_back(det);
    }

    // 判断路径：GPU 直接绘制 vs CPU 绘制后回传
    if (source_frame->is_gpu && source_frame->d_ptr) {
        // 【加速】GPU 路径：直接在 GPU 内存上绘制
        LOG_DEBUG_FMT("[GpuOSDDraw] GPU path: drawing {} boxes on GPU", filtered_dets.size());

        // 克隆 GPU 帧数据（避免修改原始帧）
        auto new_frame = std::make_shared<core::VideoFramePacket>();
        *new_frame = *source_frame;

        // 分配新的 GPU 缓冲区并复制数据
        size_t frame_size = source_frame->d_height * source_frame->d_pitch;
        void* d_output;
        CUDA_CHECK(cudaMalloc(&d_output, frame_size));
        CUDA_CHECK(cudaMemcpyAsync(d_output, source_frame->d_ptr, frame_size,
                                   cudaMemcpyDeviceToDevice, stream_));

        // GPU 绘制边界框
        if (!filtered_dets.empty()) {
            drawBoxesOnGpu(
                static_cast<unsigned char*>(d_output),
                source_frame->d_width, source_frame->d_height, source_frame->d_pitch,
                filtered_dets);
        }

        CUDA_CHECK(cudaStreamSynchronize(stream_));

        new_frame->d_ptr = d_output;
        new_frame->is_gpu = true;

        frame_count_++;
        if (snapshot_enabled_ && frame_count_ % snapshot_interval_ == 0) {
            // 需要拷贝回 CPU 保存快照
            cv::Mat cpu_mat(source_frame->d_height, source_frame->d_width, CV_8UC3);
            CUDA_CHECK(cudaMemcpy2D(cpu_mat.data, cpu_mat.step,
                                    d_output, source_frame->d_pitch,
                                    source_frame->d_width * 3, source_frame->d_height,
                                    cudaMemcpyDeviceToHost));
            auto snapshot_frame = std::make_shared<core::VideoFramePacket>();
            snapshot_frame->mat = std::make_shared<cv::Mat>(cpu_mat.clone());
            snapshot_frame->stream_id = infer_result->stream_id;
            saveSnapshot(snapshot_frame, frame_count_);
        }

        broadcast(new_frame);

    } else if (source_frame->mat && !source_frame->mat->empty()) {
        // CPU 路径：使用 OpenCV 绘制（fallback）
        LOG_DEBUG_FMT("[GpuOSDDraw] CPU path: drawing {} boxes with OpenCV", filtered_dets.size());

        auto draw_mat = std::make_shared<cv::Mat>(source_frame->source_mat->clone());

        for (const auto& det : filtered_dets) {
            cv::Rect rect(static_cast<int>(det.x), static_cast<int>(det.y),
                          static_cast<int>(det.w), static_cast<int>(det.h));
            cv::rectangle(*draw_mat, rect, box_color_, font_thickness_);

            std::string label = det.class_name;
            if (show_confidence_) {
                label += " " + std::to_string(det.confidence).substr(0, 4);
            }
            if (det.track_id >= 0) {
                label += " ID:" + std::to_string(det.track_id);
            }
            cv::putText(*draw_mat, label,
                        cv::Point(rect.x, rect.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color_, font_thickness_);
        }

        auto drawn_frame = std::make_shared<core::VideoFramePacket>();
        drawn_frame->stream_id = infer_result->stream_id;
        drawn_frame->timestamp_ms = infer_result->timestamp_ms;
        drawn_frame->mat = draw_mat;
        drawn_frame->width = draw_mat->cols;
        drawn_frame->height = draw_mat->rows;
        drawn_frame->channels = draw_mat->channels();

        frame_count_++;
        if (snapshot_enabled_ && frame_count_ % snapshot_interval_ == 0) {
            saveSnapshot(drawn_frame, frame_count_);
        }

        broadcast(drawn_frame);

    } else {
        LOG_WARN_FMT("[GpuOSDDraw] No valid frame data");
        broadcast(packet);
    }
}

void GpuOSDDrawNode::drawBoxesOnGpu(
    unsigned char* d_bgr, int width, int height, int pitch,
    const std::vector<core::InferenceResultPacket::BBox>& detections)
{
    if (detections.empty()) return;

    // 准备 GPU 数据
    int num_boxes = static_cast<int>(detections.size());
    std::vector<float> h_boxes(num_boxes * 4);
    std::vector<float> h_scores(num_boxes);
    std::vector<int> h_class_ids(num_boxes);

    for (int i = 0; i < num_boxes; ++i) {
        h_boxes[i * 4 + 0] = detections[i].x;
        h_boxes[i * 4 + 1] = detections[i].y;
        h_boxes[i * 4 + 2] = detections[i].w;
        h_boxes[i * 4 + 3] = detections[i].h;
        h_scores[i] = detections[i].confidence;
        h_class_ids[i] = detections[i].class_id;
    }

    float* d_boxes;
    float* d_scores;
    int* d_class_ids;

    CUDA_CHECK(cudaMalloc(&d_boxes, num_boxes * 4 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scores, num_boxes * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_class_ids, num_boxes * sizeof(int)));

    CUDA_CHECK(cudaMemcpyAsync(d_boxes, h_boxes.data(), num_boxes * 4 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_scores, h_scores.data(), num_boxes * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemcpyAsync(d_class_ids, h_class_ids.data(), num_boxes * sizeof(int),
                               cudaMemcpyHostToDevice, stream_));

    // 启动 CUDA 内核
    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    drawBoxesBatchKernel<<<grid, block, 0, stream_>>>(
        d_bgr, width, height, pitch,
        d_boxes, d_scores, d_class_ids,
        num_boxes, 3,  // thickness
        static_cast<unsigned char>(box_color_[0]),
        static_cast<unsigned char>(box_color_[1]),
        static_cast<unsigned char>(box_color_[2])
    );

    CUDA_CHECK(cudaStreamSynchronize(stream_));

    // 释放临时 GPU 内存
    cudaFree(d_boxes);
    cudaFree(d_scores);
    cudaFree(d_class_ids);
}

void GpuOSDDrawNode::setSnapshotEnabled(bool enabled) {
    snapshot_enabled_ = enabled;
    LOG_INFO_FMT("[GpuOSDDraw] Snapshot enabled: {}", enabled);
}

void GpuOSDDrawNode::setSnapshotInterval(int interval) {
    snapshot_interval_ = interval > 0 ? interval : 100;
    LOG_INFO_FMT("[GpuOSDDraw] Snapshot interval: {} frames", snapshot_interval_.load());
}

void GpuOSDDrawNode::setSnapshotDir(const std::string& dir) {
    snapshot_dir_ = dir;
    LOG_INFO_FMT("[GpuOSDDraw] Snapshot directory: {}", dir);
}

void GpuOSDDrawNode::saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num) {
    if (!frame || !frame->mat || frame->mat->empty()) return;
    try {
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count() % 1000000;

        std::tm tm_buf;
        localtime_r(&time_t, &tm_buf);

        std::stringstream ss;
        ss << snapshot_dir_ << "/"
           << "stream_" << frame->stream_id << "_"
           << "frame_" << std::setfill('0') << std::setw(6) << frame_num << "_"
           << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_"
           << std::setfill('0') << std::setw(6) << us
           << ".jpg";

        std::string filename = ss.str();
        bool success = cv::imwrite(filename, *frame->mat);

        if (success) {
            snapshot_count_++;
            LOG_INFO_FMT("[GpuOSDDraw] Snapshot saved: {} (frame #{}, total: {})",
                         filename, frame_num, snapshot_count_);
        }
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[GpuOSDDraw] Exception saving snapshot: {}", e.what());
    }
}

REGISTER_NODE("gpu_osd_draw", GpuOSDDrawNode)

} // namespace nodes
} // namespace ai_stream
