// src/nodes/preprocess/resize_normalize.h
#pragma once

#include "ai_stream/nodes/i_preprocess_node.h"
#include <opencv2/core/mat.hpp>

namespace ai_stream {
namespace nodes {

class ResizeNormalizeNode : public IPreprocessNode {
public:
    ResizeNormalizeNode();
    ~ResizeNormalizeNode();

    // Node 接口
    bool start() override { return true; }
    void stop() override {}
    bool isRunning() const override{return running_.load();}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    // 配置方法（可通过 JSON params 设置）
    void setTargetSize(int width, int height);
    std::pair<int, int> getTargetSize() const override;
    void setMean(const std::vector<float>& mean);
    void setStd(const std::vector<float>& std);
    void setInterpolationMethod(const std::string& method);
    void setKeepAspectRatio(bool keep_aspect_ratio);
    void setOutputDataType(const std::string& dtype);

private:
    int target_width_ = 640;
    int target_height_ = 640;
    std::vector<float> mean_{0.0f, 0.0f, 0.0f};
    std::vector<float> std_{1.0f, 1.0f, 1.0f};
    std::string interpolation_method_ = "bilinear";
    bool keep_aspect_ratio_ = false;
    std::string output_dtype_ = "float32";
};

} // namespace nodes
} // namespace ai_stream