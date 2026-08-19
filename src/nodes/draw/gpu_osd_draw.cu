// src/nodes/draw/gpu_osd_draw.cu
// GPU OSD 绘制节点——使用 ImageAcceleratorFactory，支持多后端
#include "gpu_osd_draw.h"
#include "ai_stream/core/packet.h"
#include "ai_stream/hal/image_accelerator_factory.h"
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

    bool on_border = false;

    if (py >= y && py < y + thickness && px >= x && px < x + w) on_border = true;
    if (py >= y + h - thickness && py < y + h && px >= x && px < x + w) on_border = true;
    if (px >= x && px < x + thickness && py >= y && py < y + h) on_border = true;
    if (px >= x + w - thickness && px < x + w && py >= y && py < y + h) on_border = true;

    if (on_border) {
        int idx = py * pitch + px * 3;
        image[idx + 0] = b;
        image[idx + 1] = g;
        image[idx + 2] = r;
    }
}

// Wrapper 函数
void launchDrawRectKernel(
    unsigned char* image, int width, int height, int pitch,
    int x, int y, int w, int h, int thickness,
    unsigned char b, unsigned char g, unsigned char r,
    cudaStream_t stream)
{
    dim3 block_size(16, 16);
    dim3 grid_size((width + 15) / 16, (height + 15) / 16);
    drawRectKernel<<<grid_size, block_size, 0, stream>>>(
        image, width, height, pitch,
        x, y, w, h, thickness, b, g, r);
}

} // anonymous namespace

GpuOSDDrawNode::GpuOSDDrawNode() : core::QueuedNode<IDrawNode>("GpuOSDDraw") {
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
    m_ft2_ = cv::freetype::createFreeType2();
}

GpuOSDDrawNode::~GpuOSDDrawNode() {
    stop();
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

bool GpuOSDDrawNode::onStartup() {
    if (!stream_) {
        CUDA_CHECK_BOOL(cudaStreamCreate(&stream_));
    }

    // 通过 HAL 工厂创建图像加速器
    accelerator_ = hal::ImageAcceleratorFactory::instance().create(backend_type_);
    if (!accelerator_) {
        LOG_ERROR("[GpuOSDDraw] Failed to create image accelerator");
        return false;
    }

    LOG_INFO_FMT("[GpuOSDDraw] Started (backend: {}, snapshot: {}, interval: {})",
                 accelerator_->getName(),
                 snapshot_enabled_.load(), snapshot_interval_.load());
    return true;
}

void GpuOSDDrawNode::onShutdown() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }

    LOG_INFO_FMT("[GpuOSDDraw] Stopped");
}

void GpuOSDDrawNode::drawBoxesOnGpu(
    unsigned char* d_bgr, int width, int height, int pitch,
    const std::vector<core::InferenceResultPacket::BBox>& detections) {

    // 使用 HAL 加速器绘制边界框
    std::vector<hal::BBox> hal_boxes;
    hal_boxes.reserve(detections.size());

    for (const auto& det : detections) {
        // 类别过滤
        if (!class_filter_.empty()) {
            if (std::find(class_filter_.begin(), class_filter_.end(), det.class_id) == class_filter_.end()) {
                continue;
            }
        }

        hal::BBox box;
        box.x = static_cast<float>(det.x);
        box.y = static_cast<float>(det.y);
        box.w = static_cast<float>(det.w);
        box.h = static_cast<float>(det.h);
        box.confidence = det.confidence;
        box.class_id = det.class_id;
        box.class_name = det.class_name;
        hal_boxes.push_back(box);
    }

    hal::DrawParams draw_params;
    draw_params.bgr = d_bgr;
    draw_params.width = width;
    draw_params.height = height;
    draw_params.pitch = pitch;
    draw_params.is_gpu = true;
    draw_params.stream = stream_;
    draw_params.box_color_b = static_cast<int>(box_color_[0]);
    draw_params.box_color_g = static_cast<int>(box_color_[1]);
    draw_params.box_color_r = static_cast<int>(box_color_[2]);
    draw_params.font_thickness = font_thickness_;
    draw_params.show_confidence = show_confidence_;
    draw_params.class_filter.clear(); // 已经过滤过了

    accelerator_->drawBoxes(hal_boxes, draw_params);
}

void GpuOSDDrawNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
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

    auto infer_result = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer_result) {
        LOG_ERROR_FMT("[GpuOSDDraw] Invalid inference result packet");
        return;
    }

    // 获取源帧
    auto frame = infer_result->source_frame;
    if (!frame) {
        LOG_WARN_FMT("[GpuOSDDraw] No source frame in inference result");
        broadcast(packet);
        return;
    }

    // 克隆帧以避免修改共享数据（深拷贝 CPU mat）
    auto drawn_frame = std::make_shared<core::VideoFramePacket>();
    drawn_frame->stream_id = infer_result->stream_id;
    drawn_frame->timestamp_ms = infer_result->timestamp_ms;
    drawn_frame->is_gpu = false;

    // 获取 BGR 图像数据（优先 GPU，回退 CPU）
    unsigned char* bgr_ptr = nullptr;
    int width = 0, height = 0, pitch = 0;
    bool need_gpu_to_cpu = false;

    if (frame->is_gpu && frame->d_bgr_ptr) {
        bgr_ptr = static_cast<unsigned char*>(frame->d_bgr_ptr);
        width = frame->d_bgr_width;
        height = frame->d_bgr_height;
        pitch = frame->d_bgr_pitch;
        need_gpu_to_cpu = true;
    } else if (frame->mat && !frame->mat->empty()) {
        bgr_ptr = frame->mat->data;
        width = frame->mat->cols;
        height = frame->mat->rows;
        pitch = static_cast<int>(frame->mat->step);
    }

    if (!bgr_ptr || width <= 0 || height <= 0) {
        LOG_WARN_FMT("[GpuOSDDraw] No valid BGR data available");
        broadcast(packet);
        return;
    }

    // 创建 CPU mat 用于绘制
    cv::Mat draw_img;
    if (need_gpu_to_cpu) {
        // GPU -> CPU：拷贝 GPU 数据到 CPU mat
        cv::Mat gpu_img(height, width, CV_8UC3, bgr_ptr, pitch);
        draw_img = gpu_img.clone();  // 深拷贝到 CPU
    } else if (frame->mat && !frame->mat->empty()) {
        draw_img = frame->mat->clone();  // 深拷贝避免修改原始数据
    } else {
        LOG_WARN_FMT("[GpuOSDDraw] No valid image data to draw on");
        broadcast(packet);
        return;
    }

    CUDA_CHECK(cudaEventRecord(start_event_, stream_));

    // 绘制边界框（CPU 路径）
    for (const auto& det : infer_result->detections) {
        cv::Rect rect(det.x, det.y, det.w, det.h);
        cv::rectangle(draw_img, rect, box_color_, font_thickness_);

        if (!det.class_name.empty()) {
            std::string label = det.class_name;
            if (show_confidence_) {
                label += " " + std::to_string(static_cast<int>(det.confidence * 100)) + "%";
            }
            cv::putText(draw_img, label,
                        cv::Point(det.x, det.y - 5),
                        cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color_, 1);
        }
    }

    // 绘制告警信息面板
    addPanel(draw_img, infer_result->alert_result);

    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float latency_ms = 0.0f;
    cudaEventElapsedTime(&latency_ms, start_event_, stop_event_);

    LOG_DEBUG_FMT("[GpuOSDDraw] Drew {} boxes on {}x{} (CPU), latency={:.2f}ms",
                  infer_result->detections.size(), width, height, latency_ms);

    // 设置绘制后的帧数据
    drawn_frame->mat = std::make_shared<cv::Mat>(std::move(draw_img));
    drawn_frame->width = drawn_frame->mat->cols;
    drawn_frame->height = drawn_frame->mat->rows;
    drawn_frame->channels = drawn_frame->mat->channels();

    // 快照功能
    frame_count_++;
    if (snapshot_enabled_ && frame_count_ % snapshot_interval_ == 0) {
        saveSnapshot(drawn_frame, frame_count_);
    }

    // 广播 DECODED_FRAME 包（供 RTMP sink 和 evidence node 使用）
    broadcast(drawn_frame);
}

void GpuOSDDrawNode::setSnapshotEnabled(bool enabled) {
    snapshot_enabled_ = enabled;
}

void GpuOSDDrawNode::setSnapshotInterval(int interval) {
    snapshot_interval_ = interval;
}

void GpuOSDDrawNode::setSnapshotDir(const std::string& dir) {
    snapshot_dir_ = dir;
}

void GpuOSDDrawNode::saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num) {
    if (!frame->mat || frame->mat->empty()) return;

    std::stringstream ss;
    ss << snapshot_dir_ << "/snapshot_" << std::setfill('0') << std::setw(6) << frame_num << ".jpg";
    try {
        cv::imwrite(ss.str(), *frame->mat);
        snapshot_count_++;
        LOG_INFO_FMT("[GpuOSDDraw] Saved snapshot: {}", ss.str());
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[GpuOSDDraw] Failed to save snapshot: {}", e.what());
    }
}

REGISTER_NODE("gpu_osd_draw", GpuOSDDrawNode)

} // namespace nodes
} // namespace ai_stream
