// src/nodes/draw/osd_draw.h
#pragma once

#include "ai_stream/nodes/i_draw_node.h"
#include <opencv2/core/mat.hpp>

namespace ai_stream {
namespace nodes {

// COCO 骨架连线定义（基于关键点索引）
const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1},   // nose -> left_eye
    {0, 2},   // nose -> right_eye
    {1, 3},   // left_eye -> left_ear
    {2, 4},   // right_eye -> right_ear
    {5, 6},   // left_shoulder -> right_shoulder
    {5, 7},   // left_shoulder -> left_elbow
    {7, 9},   // left_elbow -> left_wrist
    {6, 8},   // right_shoulder -> right_elbow
    {8, 10},  // right_elbow -> right_wrist
    {5, 11},  // left_shoulder -> left_hip
    {6, 12},  // right_shoulder -> right_hip
    {11, 12}, // left_hip -> right_hip
    {11, 13}, // left_hip -> left_knee
    {13, 15}, // left_knee -> left_ankle
    {12, 14}, // right_hip -> right_knee
    {14, 16}  // right_knee -> right_ankle
};

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