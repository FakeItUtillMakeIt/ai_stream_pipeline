// src/nodes/draw/osd_draw.h
#pragma once

#include "ai_stream/nodes/i_draw_node.h"
#include "ai_stream/core/queued_node.h"
#include "ai_stream/hal/i_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include <opencv2/core/mat.hpp>
#ifdef HAVE_OPENCV_FREETYPE
#include <opencv2/freetype.hpp>
#endif

namespace ai_stream {
namespace nodes {

/**
 * @brief OSD 绘制节点（统一 CPU/GPU 路径）
 *
 * 绘制架构：几何绘制（矩形框）通过 HAL IImageAccelerator::drawBoxes 路由，
 * 按数据位置与后端能力自动选择：
 * - source_frame 带设备端 BGR（d_bgr_ptr）且 NPP 可用 → GPU 画框，D2H 后节点层补文字
 * - 否则 → CPU 后端画框
 * 文本标签 / 关键点骨架 / 告警面板始终在节点层 CPU 绘制
 * （GPU 无 glyph 光栅化，且绘制量小、依赖字体）。
 */
class OSDDrawNode : public core::QueuedNode<IDrawNode> {
public:
    OSDDrawNode();
    ~OSDDrawNode() override;

    // 消费 InferenceResultPacket 时需要保住 source_frame 的设备端 BGR
    // （d_bgr_ptr），供 GPU 绘制路径使用。
    bool acceptsGpuFrame() const override { return true; }

    // IDrawNode 接口
    void setBoxColor(int b, int g, int r) override;
    void setFontThickness(int thickness) override;
    void setShowConfidence(bool show) override;
    void setClassFilter(const std::vector<int>& class_ids) override;

    // 设置图像加速器后端（AUTO 时按数据位置自动选择）
    void setImageAcceleratorBackend(hal::ImageAcceleratorBackend backend) { backend_type_ = backend; }

    // QueuedNode 接口
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;
    bool onStartup() override;
    void setSnapshotEnabled(bool enabled) override;
    void setSnapshotInterval(int interval) override;
    void setSnapshotDir(const std::string& dir) override;

private:
    // 设备端 BGR 画框 + D2H，成功时填充 draw_img 并返回 true
    bool drawGpu(const std::shared_ptr<core::VideoFramePacket>& frame,
                 const std::shared_ptr<core::InferenceResultPacket>& infer_result,
                 cv::Mat& draw_img);

    // CPU 底图画框（HAL CPU 后端），成功时填充 draw_img 并返回 true
    bool drawCpu(const std::shared_ptr<core::VideoFramePacket>& frame,
                 const std::shared_ptr<core::InferenceResultPacket>& infer_result,
                 cv::Mat& draw_img);

    // 节点层 CPU 叠加：标签（含 track id）、关键点与骨架
    void drawOverlayCpu(cv::Mat& img,
                        const std::vector<core::InferenceResultPacket::BBox>& detections);

    // hal::BBox 转换（应用类别过滤）
    std::vector<hal::BBox> toHalBoxes(
        const std::vector<core::InferenceResultPacket::BBox>& detections) const;

    hal::IImageAccelerator* getCpuAccelerator();
#ifdef WITH_CUDA
    hal::IImageAccelerator* getGpuAccelerator();
#endif

    void saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num);

private:
    cv::Scalar box_color_{0, 255, 0}; // 默认绿色 BGR
    int font_thickness_ = 1;
    bool show_confidence_ = true;
    std::vector<int> class_filter_;
    // 快照配置
    std::atomic<bool> snapshot_enabled_{false};
    std::atomic<int> snapshot_interval_{100};
    std::string snapshot_dir_ = "./snapshots";
    int snapshot_count_ = 0;
    int frame_count_ = 0;

    // HAL 图像加速器
    hal::ImageAcceleratorBackend backend_type_ = hal::ImageAcceleratorBackend::AUTO;
    hal::ImageAcceleratorPtr cpu_accelerator_;
#ifdef WITH_CUDA
    hal::ImageAcceleratorPtr gpu_accelerator_;
    bool gpu_backend_checked_ = false;
    void* cuda_stream_ = nullptr;
    bool owns_cuda_stream_ = false;
#endif
};

} // namespace nodes
} // namespace ai_stream
