// src/nodes/draw/osd_draw.h
#pragma once

#include "ai_stream/nodes/i_draw_node.h"
#include <opencv2/core/mat.hpp>

namespace ai_stream {
namespace nodes {

class OSDDrawNode : public IDrawNode {
public:
    OSDDrawNode();
    ~OSDDrawNode() override = default;

    // IDrawNode 接口
    void setBoxColor(int b, int g, int r) override;
    void setFontThickness(int thickness) override;
    void setShowConfidence(bool show) override;
    void setClassFilter(const std::vector<int>& class_ids) override;

    // Node 接口
    bool start() override;
    void stop() override {}
    bool isRunning() const override{return running_.load();}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;
    void setSnapshotEnabled(bool enabled) override;
    void setSnapshotInterval(int interval) override;
    void setSnapshotDir(const std::string& dir) override;

private:
    void saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num);
    void addPanel(cv::Mat& img,const std::vector<rules::AlertResult>& alert_results);
private:
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