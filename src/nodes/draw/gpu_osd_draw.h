// src/nodes/draw/gpu_osd_draw.h
// 【加速优化】GPU 加速 OSD 绘制
#pragma once

#include "ai_stream/nodes/i_draw_node.h"
#include <opencv2/core/mat.hpp>
#include <cuda_runtime.h>
#include <opencv2/freetype.hpp>

namespace ai_stream {
namespace nodes {

/**
 * @brief GPU 加速的 OSD 绘制节点
 *
 * 使用 CUDA 在 GPU 上直接绘制边界框和文本，避免 CPU-GPU 数据传输。
 * 适合高帧率、高分辨率的视频流处理。
 */
class GpuOSDDrawNode : public IDrawNode {
public:
    GpuOSDDrawNode();
    ~GpuOSDDrawNode() override;

    // IDrawNode 接口
    void setBoxColor(int b, int g, int r) override;
    void setFontThickness(int thickness) override;
    void setShowConfidence(bool show) override;
    void setClassFilter(const std::vector<int>& class_ids) override;

    // Node 接口
    bool start() override;
    void stop() override;
    bool isRunning() const override { return running_.load(); }
    void pushData(std::shared_ptr<core::BasePacket> packet) override;
    void setSnapshotEnabled(bool enabled) override;
    void setSnapshotInterval(int interval) override;
    void setSnapshotDir(const std::string& dir) override;
    void setFontFile(const std::string& font_path) override;
    void setLogoFile(const std::string& logo_path) override;

    // GPU 特定配置
    void setGpuDeviceId(int device_id) { device_id_ = device_id; }
    int getGpuDeviceId() const { return device_id_; }

private:
    void saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num);

    // CUDA 绘制内核
    void drawBoxesOnGpu(
        unsigned char* d_bgr, int width, int height, int pitch,
        const std::vector<core::InferenceResultPacket::BBox>& detections);

    int device_id_ = 0;
    cudaStream_t stream_ = nullptr;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

    cv::Scalar box_color_{0, 255, 0}; // 默认绿色 BGR
    int font_thickness_ = 2;
    bool show_confidence_ = true;
    std::vector<int> class_filter_;

    // 快照配置
    std::atomic<bool> snapshot_enabled_{false};
    std::atomic<int> snapshot_interval_{100};
    std::string snapshot_dir_ = "./snapshots";
    int snapshot_count_ = 0;
    int frame_count_ = 0;
};

} // namespace nodes
} // namespace ai_stream
