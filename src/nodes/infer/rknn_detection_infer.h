// src/nodes/infer/rknn_detection_infer.h
// RKNN 检测推理节点——使用 HAL 抽象接口
#pragma once

#include "ai_stream/nodes/i_infer_node.h"
#include "ai_stream/core/bounded_queue.h"
#include "ai_stream/hal/i_inference_engine.h"

#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <deque>

namespace ai_stream {
namespace nodes {

/**
 * @brief RKNN 检测推理节点
 *
 * 使用 HAL 推理引擎抽象接口，支持 RKNN/TensorRT/ASCEND 后端。
 * 通过 InferenceEngineFactory 创建具体后端，无需编译时依赖特定 SDK。
 */
class RknnDetectionInferNode : public IInferNode {
public:
    RknnDetectionInferNode();
    ~RknnDetectionInferNode() override;

    bool loadModel(const std::string& model_path) override;
    void setPrecision(const std::string& precision) override;
    void setBatchSize(int batch_size) override;
    void setInputSize(int width, int height) override;
    std::pair<int, int> getInputSize() const override;
    void setClassNames(const std::vector<std::string>& names) override;
    std::vector<std::string> getClassNames() const override;

    void setDetectorType(DetectorType type) override;
    DetectorType getDetectorType() const override;

    bool start() override;
    void stop() override;
    bool isRunning() const override { return running_.load(); }
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void inferLoop();
    void preprocess(const cv::Mat& image, float* buffer);
    std::vector<core::InferenceResultPacket::BBox> postprocess(
        const float* output, int output_size, float conf_thresh);

    // 使用抽象推理引擎
    hal::InferenceEnginePtr engine_;
    hal::InferenceBackend backend_type_ = hal::InferenceBackend::AUTO;

    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 1;
    std::string precision_ = "fp16";
    DetectorType detector_type_ = DetectorType::DETECTION;

    std::vector<std::string> class_names_;

    // 数据队列和线程
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> queue_{64};
    std::thread worker_;
    std::atomic<bool> running_{false};

    std::chrono::milliseconds batch_timeout_ms_{20};
    std::atomic<int> max_batch_size_{1};
};

} // namespace nodes
} // namespace ai_stream
