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
    bool start() override { return true; }
    void stop() override {}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    cv::Scalar box_color_{0, 255, 0}; // 默认绿色 BGR
    int font_thickness_ = 2;
    bool show_confidence_ = true;
    std::vector<int> class_filter_;
};

} // namespace nodes
} // namespace ai_stream