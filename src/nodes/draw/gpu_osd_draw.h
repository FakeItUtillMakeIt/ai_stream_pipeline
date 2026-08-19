// src/nodes/draw/gpu_osd_draw.h
// GPU OSD 绘制节点——使用 ImageAcceleratorFactory，支持多后端
#pragma once

#include "ai_stream/nodes/i_draw_node.h"
#include "ai_stream/core/queued_node.h"
#include "ai_stream/hal/i_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include <cuda_runtime.h>
#include <opencv2/core.hpp>
#include <atomic>

namespace ai_stream {
namespace nodes {

class GpuOSDDrawNode : public core::QueuedNode<IDrawNode> {
public:
    GpuOSDDrawNode();
    ~GpuOSDDrawNode() override;

    // QueuedNode 接口
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;
    bool onStartup() override;
    void onShutdown() override;

    // 配置接口
    void setBoxColor(int b, int g, int r);
    void setFontThickness(int thickness);
    void setShowConfidence(bool show);
    void setClassFilter(const std::vector<int>& class_ids);

    // 快照功能
    void setSnapshotEnabled(bool enabled);
    void setSnapshotInterval(int interval);
    void setSnapshotDir(const std::string& dir);
    int getSnapshotCount() const { return snapshot_count_.load(); }

    // 设置图像加速器后端类型
    void setImageAcceleratorBackend(hal::ImageAcceleratorBackend backend) { backend_type_ = backend; }

private:
    void drawBoxesOnGpu(
        unsigned char* d_bgr, int width, int height, int pitch,
        const std::vector<core::InferenceResultPacket::BBox>& detections);

    void saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num);

    // HAL 图像加速器
    hal::ImageAcceleratorPtr accelerator_;
    hal::ImageAcceleratorBackend backend_type_ = hal::ImageAcceleratorBackend::AUTO;

    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_ = nullptr;
    cudaEvent_t stop_event_ = nullptr;

    // 配置
    cv::Scalar box_color_ = cv::Scalar(0, 255, 0);
    int font_thickness_ = 2;
    bool show_confidence_ = true;
    std::vector<int> class_filter_;

    // 快照
    std::atomic<bool> snapshot_enabled_{false};
    std::atomic<int> snapshot_interval_{30};
    std::atomic<int> snapshot_count_{0};
    std::atomic<int> frame_count_{0};
    std::string snapshot_dir_ = "./snapshots";
};

} // namespace nodes
} // namespace ai_stream
