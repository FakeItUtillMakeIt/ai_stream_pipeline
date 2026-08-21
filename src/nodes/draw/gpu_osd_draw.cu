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
#ifdef HAVE_OPENCV_FREETYPE
    m_ft2_ = cv::freetype::createFreeType2();
#endif
}

GpuOSDDrawNode::~GpuOSDDrawNode() {
    stop();
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
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

    // 释放加速器
    accelerator_.reset();

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
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 workerLoop 中检查，worker 线程会自然退出
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

    // 克隆帧以避免修改共享数据
    auto drawn_frame = std::make_shared<core::VideoFramePacket>();
    drawn_frame->stream_id = infer_result->stream_id;
    drawn_frame->timestamp_ms = infer_result->timestamp_ms;
    drawn_frame->is_gpu = false;

    CUDA_CHECK(cudaEventRecord(start_event_, stream_));

    // 获取 BGR 图像数据
    unsigned char* bgr_ptr = nullptr;
    int width = 0, height = 0, pitch = 0;
    bool src_is_gpu = false;

    if (frame->is_gpu && frame->d_bgr_ptr) {
        bgr_ptr = static_cast<unsigned char*>(frame->d_bgr_ptr);
        width = frame->d_bgr_width;
        height = frame->d_bgr_height;
        pitch = frame->d_bgr_pitch;
        src_is_gpu = true;
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

    cv::Mat draw_img;

    if (src_is_gpu) {
        // GPU 路径：分配新的 GPU 缓冲区，拷贝数据，绘制，然后拷贝回 CPU
        size_t gpu_size = static_cast<size_t>(pitch) * height;
        unsigned char* d_clone = nullptr;

        cudaError_t err = cudaMalloc(&d_clone, gpu_size);
        if (err != cudaSuccess) {
            LOG_ERROR_FMT("[GpuOSDDraw] cudaMalloc failed: {}", cudaGetErrorString(err));
            broadcast(packet);
            return;
        }

        err = cudaMemcpyAsync(d_clone, bgr_ptr, gpu_size, cudaMemcpyDeviceToDevice, stream_);
        if (err != cudaSuccess) {
            LOG_ERROR_FMT("[GpuOSDDraw] cudaMemcpyAsync failed: {}", cudaGetErrorString(err));
            cudaFree(d_clone);
            broadcast(packet);
            return;
        }

        // 使用 HAL 加速器在 GPU 上绘制边界框
        if (!infer_result->detections.empty()) {
            drawBoxesOnGpu(d_clone, width, height, pitch, infer_result->detections);
        }

        // 拷贝 GPU 数据到 CPU mat
        draw_img = cv::Mat(height, width, CV_8UC3);
        err = cudaMemcpyAsync(draw_img.data, d_clone, gpu_size, cudaMemcpyDeviceToHost, stream_);
        if (err != cudaSuccess) {
            LOG_ERROR_FMT("[GpuOSDDraw] cudaMemcpyAsync(D2H) failed: {}", cudaGetErrorString(err));
            cudaFree(d_clone);
            broadcast(packet);
            return;
        }

        cudaStreamSynchronize(stream_);

        // 释放 GPU 缓冲区
        cudaFree(d_clone);
    } else {
        // CPU 路径：深拷贝避免修改原始数据
        draw_img = frame->mat->clone();

        // 使用 OpenCV 绘制边界框
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
    }

    // 绘制告警信息面板（始终在 CPU 上）
    addPanel(draw_img, infer_result->alert_result);

    CUDA_CHECK(cudaEventRecord(stop_event_, stream_));
    CUDA_CHECK(cudaEventSynchronize(stop_event_));

    float latency_ms = 0.0f;
    cudaEventElapsedTime(&latency_ms, start_event_, stop_event_);

    LOG_DEBUG_FMT("[GpuOSDDraw] Drew {} boxes on {}x{} ({}), latency={:.2f}ms",
                  infer_result->detections.size(), width, height,
                  src_is_gpu ? "GPU" : "CPU", latency_ms);

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
