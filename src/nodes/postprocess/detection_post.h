// src/nodes/postprocess/detection_post.h
#pragma once

#include "ai_stream/core/node.h"

namespace ai_stream {
namespace nodes {

class DetectionPostProcessNode : public core::Node {
public:
    DetectionPostProcessNode();
    ~DetectionPostProcessNode() override = default;

    bool start() override { return true; }
    void stop() override {}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    // 配置参数
    void setConfidenceThreshold(float threshold) { conf_thresh_ = threshold; }
    void setNmsThreshold(float threshold) { nms_thresh_ = threshold; }

private:
    float conf_thresh_ = 0.5f;
    float nms_thresh_ = 0.4f;
};

} // namespace nodes
} // namespace ai_stream