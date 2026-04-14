// src/nodes/preprocess/resize_normalize.h
#pragma once

#include "ai_stream/core/node.h"
#include <opencv2/core/mat.hpp>

namespace ai_stream {
namespace nodes {

class ResizeNormalizeNode : public core::Node {
public:
    ResizeNormalizeNode();
    ~ResizeNormalizeNode() override = default;

    // Node 接口
    bool start() override { return true; }
    void stop() override {}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    // 配置方法（可通过 JSON params 设置）
    void setTargetSize(int width, int height);
    void setMean(const std::vector<float>& mean);
    void setStd(const std::vector<float>& std);

private:
    int target_width_ = 640;
    int target_height_ = 640;
    std::vector<float> mean_{0.0f, 0.0f, 0.0f};
    std::vector<float> std_{1.0f, 1.0f, 1.0f};
};

} // namespace nodes
} // namespace ai_stream