// src/nodes/draw/osd_draw.cpp
// OSD 绘制节点——矩形框经 HAL IImageAccelerator 路由（GPU 数据走 NPP，
// CPU 数据走 CPU/RGA/DVPP 后端），文字/关键点/告警面板在节点层绘制。
#include "utils/time_util.h"
#include "osd_draw.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm>
#include <fstream>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#include "ai_stream/hal/gpu_buffer_pool.h"
#endif

namespace ai_stream {
namespace nodes {

OSDDrawNode::OSDDrawNode() : core::QueuedNode<IDrawNode>("OSDDraw") {
#ifdef HAVE_OPENCV_FREETYPE
    m_ft2_ = cv::freetype::createFreeType2();
#endif
}

OSDDrawNode::~OSDDrawNode() {
    stop();
    // STREAM_END may have already cleared running_ before destruction, so
    // ensure resources are released even when QueuedNode::stop() is a no-op.
#ifdef WITH_CUDA
    if (cuda_stream_ && owns_cuda_stream_) {
        cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_));
        cudaStreamDestroy(static_cast<cudaStream_t>(cuda_stream_));
    }
    cuda_stream_ = nullptr;
    owns_cuda_stream_ = false;
    gpu_accelerator_.reset();
    gpu_backend_checked_ = false;
#endif
    cpu_accelerator_.reset();
    LOG_INFO_FMT("[OSDDraw] Destroyed (frames: {}, snapshots: {})",
                 frame_count_, snapshot_count_);
}

void OSDDrawNode::setBoxColor(int b, int g, int r) {
    box_color_ = cv::Scalar(b, g, r);
}

void OSDDrawNode::setFontThickness(int thickness) {
    font_thickness_ = thickness;
}

void OSDDrawNode::setShowConfidence(bool show) {
    show_confidence_ = show;
}

void OSDDrawNode::setClassFilter(const std::vector<int>& class_ids) {
    class_filter_ = class_ids;
}

hal::IImageAccelerator* OSDDrawNode::getCpuAccelerator() {
    if (cpu_accelerator_) {
        return cpu_accelerator_.get();
    }

    // CPU 路径处理的是主机内存指针，NPP 后端只能操作设备内存——
    // 绝不能进入候选（否则 CUDA kernel 读 host 指针会毒化整个 context）。
    // AUTO 时 RGA/DVPP 仅在对应平台编译，x86 上自然回退 CPU。
    std::vector<hal::ImageAcceleratorBackend> candidates;
    switch (backend_type_) {
    case hal::ImageAcceleratorBackend::RGA:
        candidates = {hal::ImageAcceleratorBackend::RGA, hal::ImageAcceleratorBackend::CPU};
        break;
    case hal::ImageAcceleratorBackend::DVPP:
        candidates = {hal::ImageAcceleratorBackend::DVPP, hal::ImageAcceleratorBackend::CPU};
        break;
    case hal::ImageAcceleratorBackend::NPP:
        LOG_WARN("[OSDDraw] NPP cannot draw on host memory; using CPU backend");
        candidates = {hal::ImageAcceleratorBackend::CPU};
        break;
    case hal::ImageAcceleratorBackend::CPU:
    case hal::ImageAcceleratorBackend::AUTO:
    default:
        candidates = {hal::ImageAcceleratorBackend::CPU};
        break;
    }

    for (auto backend : candidates) {
        auto accel = hal::ImageAcceleratorFactory::instance().create(backend);
        if (accel) {
            LOG_INFO_FMT("[OSDDraw] CPU draw backend: {}", accel->getName());
            cpu_accelerator_ = std::move(accel);
            return cpu_accelerator_.get();
        }
    }

    LOG_ERROR("[OSDDraw] Failed to create CPU image accelerator");
    return nullptr;
}

#ifdef WITH_CUDA
hal::IImageAccelerator* OSDDrawNode::getGpuAccelerator() {
    if (gpu_backend_checked_) {
        return gpu_accelerator_.get();
    }
    gpu_backend_checked_ = true;
    auto accel = hal::ImageAcceleratorFactory::instance().create(
        hal::ImageAcceleratorBackend::NPP);
    if (!accel) {
        LOG_WARN("[OSDDraw] NPP backend not available, GPU draw disabled");
        return nullptr;
    }
    gpu_accelerator_ = std::move(accel);
    LOG_INFO_FMT("[OSDDraw] GPU draw backend: {}", gpu_accelerator_->getName());
    return gpu_accelerator_.get();
}
#endif

bool OSDDrawNode::onStartup()
{
#ifdef WITH_CUDA
    if (!cuda_stream_) {
        cudaStream_t stream = nullptr;
        if (cudaStreamCreate(&stream) != cudaSuccess) {
            LOG_WARN("[OSDDraw] cudaStreamCreate failed, GPU draw disabled");
        } else {
            cuda_stream_ = stream;
            owns_cuda_stream_ = true;
        }
    }
#endif

    // 如果启用了快照，创建保存目录
    if (snapshot_enabled_) {
        if (!std::filesystem::exists(snapshot_dir_)) {
            try {
                std::filesystem::create_directories(snapshot_dir_);
                LOG_INFO_FMT("[OSDDraw] Created snapshot directory: {}", snapshot_dir_);
            } catch (const std::exception& e) {
                LOG_ERROR_FMT("[OSDDraw] Failed to create snapshot directory: {}", e.what());
                snapshot_enabled_ = false;
            }
        }
    }

    LOG_INFO_FMT("[OSDDraw] Started (snapshot: {}, interval: {})",
                 snapshot_enabled_.load(), snapshot_interval_.load());
    return true;
}

std::vector<hal::BBox> OSDDrawNode::toHalBoxes(
    const std::vector<core::InferenceResultPacket::BBox>& detections) const {
    std::vector<hal::BBox> hal_boxes;
    hal_boxes.reserve(detections.size());
    for (const auto& det : detections) {
        if (!class_filter_.empty() &&
            std::find(class_filter_.begin(), class_filter_.end(), det.class_id) == class_filter_.end()) {
            continue;
        }
        hal::BBox box;
        box.x = det.x;
        box.y = det.y;
        box.w = det.w;
        box.h = det.h;
        box.confidence = det.confidence;
        box.class_id = det.class_id;
        box.class_name = det.class_name;
        hal_boxes.push_back(box);
    }
    return hal_boxes;
}

#ifdef WITH_CUDA
bool OSDDrawNode::drawGpu(const std::shared_ptr<core::VideoFramePacket>& frame,
                          const std::shared_ptr<core::InferenceResultPacket>& infer_result,
                          cv::Mat& draw_img) {
    if (!frame->d_bgr_ptr) return false;

    auto* accelerator = getGpuAccelerator();
    if (!accelerator || !cuda_stream_) return false;

    const int width = frame->d_bgr_width;
    const int height = frame->d_bgr_height;
    if (width <= 0 || height <= 0) return false;
    const int pitch = frame->d_bgr_pitch > 0 ? frame->d_bgr_pitch : width * 3;
    const size_t gpu_bytes = static_cast<size_t>(height) * static_cast<size_t>(pitch);

    // 池化 D2D 克隆，避免修改共享的源帧缓冲区
    auto d_clone = hal::GpuBufferPool::instance().acquire(gpu_bytes);
    if (!d_clone) return false;

    auto stream = static_cast<cudaStream_t>(cuda_stream_);
    if (cudaMemcpyAsync(d_clone.get(), frame->d_bgr_ptr, gpu_bytes,
                        cudaMemcpyDeviceToDevice, stream) != cudaSuccess) {
        LOG_ERROR("[OSDDraw] GPU draw: D2D copy failed");
        return false;
    }

    auto hal_boxes = toHalBoxes(infer_result->detections);
    if (!hal_boxes.empty()) {
        hal::DrawParams params;
        params.bgr = static_cast<uint8_t*>(d_clone.get());
        params.width = width;
        params.height = height;
        params.pitch = static_cast<size_t>(pitch);
        params.is_gpu = true;
        params.stream = cuda_stream_;
        params.box_color_b = static_cast<int>(box_color_[0]);
        params.box_color_g = static_cast<int>(box_color_[1]);
        params.box_color_r = static_cast<int>(box_color_[2]);
        params.font_thickness = font_thickness_;
        params.show_confidence = show_confidence_;
        params.draw_labels = false;  // 文本由节点层在 D2H 后绘制
        if (!accelerator->drawBoxes(hal_boxes, params)) {
            LOG_WARN("[OSDDraw] GPU drawBoxes failed, falling back to CPU path");
            return false;
        }
    }

    draw_img = cv::Mat(height, width, CV_8UC3);
    if (cudaMemcpy2DAsync(draw_img.data, width * 3,
                          d_clone.get(), pitch,
                          width * 3, height,
                          cudaMemcpyDeviceToHost, stream) != cudaSuccess) {
        LOG_ERROR("[OSDDraw] GPU draw: D2H copy failed");
        return false;
    }
    if (cudaStreamSynchronize(stream) != cudaSuccess) {
        LOG_ERROR("[OSDDraw] GPU draw: stream sync failed");
        return false;
    }

    // 标签 / 关键点 / track id 在 CPU 上补齐
    drawOverlayCpu(draw_img, infer_result->detections);
    return true;
}
#endif

bool OSDDrawNode::drawCpu(const std::shared_ptr<core::VideoFramePacket>& frame,
                          const std::shared_ptr<core::InferenceResultPacket>& infer_result,
                          cv::Mat& draw_img) {
    // preprocess 输出的 mat 可能是 CV_32FC3（供推理使用），不能作为绘制
    // 底图；优先使用保留的原始 BGR 帧 source_mat。
    const cv::Mat* cpu_bgr = frame->mat.get();
    if ((!cpu_bgr || cpu_bgr->empty() || cpu_bgr->type() != CV_8UC3) &&
        frame->source_mat && !frame->source_mat->empty() &&
        frame->source_mat->type() == CV_8UC3) {
        cpu_bgr = frame->source_mat.get();
    }
    if (!cpu_bgr || cpu_bgr->empty() || cpu_bgr->type() != CV_8UC3) {
        LOG_WARN("[OSDDraw] No valid CV_8UC3 CPU BGR frame");
        return false;
    }

    draw_img = cpu_bgr->clone();

    // 矩形框经 HAL 绘制（后端可为 CPU / RGA / DVPP），标签由节点层统一画
    if (auto* accelerator = getCpuAccelerator()) {
        hal::DrawParams params;
        params.bgr = draw_img.data;
        params.width = draw_img.cols;
        params.height = draw_img.rows;
        params.pitch = static_cast<size_t>(draw_img.step);
        params.is_gpu = false;
        params.box_color_b = static_cast<int>(box_color_[0]);
        params.box_color_g = static_cast<int>(box_color_[1]);
        params.box_color_r = static_cast<int>(box_color_[2]);
        params.font_thickness = font_thickness_;
        params.show_confidence = show_confidence_;
        params.draw_labels = false;
        if (!accelerator->drawBoxes(toHalBoxes(infer_result->detections), params)) {
            LOG_WARN("[OSDDraw] HAL drawBoxes failed on CPU path");
        }
    }

    drawOverlayCpu(draw_img, infer_result->detections);
    return true;
}

void OSDDrawNode::drawOverlayCpu(cv::Mat& img,
                                 const std::vector<core::InferenceResultPacket::BBox>& detections) {
    for (const auto& det : detections) {
        // 类别过滤
        if (!class_filter_.empty() &&
            std::find(class_filter_.begin(), class_filter_.end(), det.class_id) == class_filter_.end()) {
            continue;
        }

        cv::Rect rect(static_cast<int>(det.x), static_cast<int>(det.y),
                      static_cast<int>(det.w), static_cast<int>(det.h));

        if (det.has_keypoints)
        {
            // 1. 先画骨架连线（先画线，再画点，避免点盖住线）
            for (const auto& [i, j] : core::SKELETON)
            {
                const auto& kp1 = det.keypoints[i];
                const auto& kp2 = det.keypoints[j];

                // 两个点都可见才连线
                if (kp1.visible && kp2.visible)
                {
                    cv::Point p1(static_cast<int>(kp1.x), static_cast<int>(kp1.y));
                    cv::Point p2(static_cast<int>(kp2.x), static_cast<int>(kp2.y));
                    cv::line(img, p1, p2, cv::Scalar(255, 128, 0), 1, cv::LINE_AA);
                }
            }

            // 2. 再画关键点（点和序号）
            for (int k = 0; k < 17; ++k)
            {
                const auto& kp = det.keypoints[k];
                if (!kp.visible) continue;  // 不可见跳过，或画灰色点

                cv::Point pt(static_cast<int>(kp.x), static_cast<int>(kp.y));

                // 画实心圆
                cv::circle(img, pt, 5, cv::Scalar(0, 255, 0), -1);
                // 白边
                cv::circle(img, pt, 5, cv::Scalar(255, 255, 255), 1);

                // 标注序号（小字体）
                cv::putText(img, std::to_string(k),
                            cv::Point(pt.x + 8, pt.y - 8),
                            cv::FONT_HERSHEY_SIMPLEX, 0.5,
                            cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
            }
        }

        std::string label = det.class_name;
        if (show_confidence_) {
            label += " " + std::to_string(det.confidence).substr(0, 4);
        }
        if (det.track_id >= 0) {
            label += " ID:" + std::to_string(det.track_id);
        }
        cv::putText(img, label,
                    cv::Point(rect.x, rect.y - 5),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, box_color_, font_thickness_);
    }
}

void OSDDrawNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[OSDDraw] Received stream end");
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 workerLoop 中检查，worker 线程会自然退出
        broadcast(packet);
        return;
    }
    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }

    auto infer_result = std::static_pointer_cast<core::InferenceResultPacket>(packet);
    auto frame = infer_result->source_frame;
    if (!frame) {
        LOG_WARN("[OSDDraw] Inference result missing source frame");
        broadcast(packet);
        return;
    }

    cv::Mat draw_img;
    bool drawn = false;
#ifdef WITH_CUDA
    if (frame->d_bgr_ptr) {
        drawn = drawGpu(frame, infer_result, draw_img);
        if (!drawn) {
            LOG_DEBUG("[OSDDraw] GPU draw unavailable, falling back to CPU path");
        }
    }
#endif
    if (!drawn) {
        drawn = drawCpu(frame, infer_result, draw_img);
    }
    if (!drawn) {
        broadcast(packet);
        return;
    }

    // 绘制告警信息面板（始终在 CPU 上）
    addPanel(draw_img, infer_result->alert_result);

    // 构造新的视频帧包（包含绘制后的图像）
    auto drawn_frame = std::make_shared<core::VideoFramePacket>();
    drawn_frame->stream_id = infer_result->stream_id;
    drawn_frame->timestamp_ms = infer_result->timestamp_ms;
    drawn_frame->mat = std::make_shared<cv::Mat>(std::move(draw_img));
    drawn_frame->width = drawn_frame->mat->cols;
    drawn_frame->height = drawn_frame->mat->rows;
    drawn_frame->channels = drawn_frame->mat->channels();

    frame_count_++;
    if (snapshot_enabled_ && frame_count_ % snapshot_interval_ == 0) {
        saveSnapshot(drawn_frame, frame_count_);
    }

    broadcast(drawn_frame);
}

void OSDDrawNode::setSnapshotEnabled(bool enabled)
{
    snapshot_enabled_ = enabled;
    LOG_INFO_FMT("[OSDDraw] Snapshot enabled: {}", enabled);
}

void OSDDrawNode::setSnapshotInterval(int interval)
{
    snapshot_interval_ = interval > 0 ? interval : 100;
    LOG_INFO_FMT("[OSDDraw] Snapshot interval: {} frames", snapshot_interval_.load());
}

void OSDDrawNode::setSnapshotDir(const std::string& dir)
{
    snapshot_dir_ = dir;
    LOG_INFO_FMT("[OSDDraw] Snapshot directory: {}", dir);
}

void OSDDrawNode::saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num)
{
    if(!frame || !frame->mat || frame->mat->empty())
        return;
    try {
        // 生成文件名
        auto now = std::chrono::system_clock::now();
        auto time_t = std::chrono::system_clock::to_time_t(now);
        auto us = std::chrono::duration_cast<std::chrono::microseconds>(
            now.time_since_epoch()).count() % 1000000;

        std::tm tm_buf = utils::TimeUtil::safeLocaltime(time_t);

        std::stringstream ss;
        ss << snapshot_dir_ << "/"
           << "stream_" << frame->stream_id << "_"
           << "frame_" << std::setfill('0') << std::setw(6) << frame_num << "_"
           << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << "_"
           << std::setfill('0') << std::setw(6) << us
           << ".jpg";

        std::string filename = ss.str();

        // 保存图片
        bool success = cv::imwrite(filename, *frame->mat);

        if (success) {
            snapshot_count_++;
            LOG_INFO_FMT("[OSDDraw] Snapshot saved: {} (frame #{}, total: {})",
                         filename, frame_num, snapshot_count_);
        }
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[OSDDraw] Exception saving snapshot: {}", e.what());
    }
}

REGISTER_NODE("osd_draw", OSDDrawNode)

} // namespace nodes
} // namespace ai_stream
