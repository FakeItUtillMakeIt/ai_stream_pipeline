// src/nodes/infer/pose_infer.h

#pragma once
#include "ai_stream/nodes/i_infer_node.h"
#include "src/core/frame_queue.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <deque>
#include <opencv2/core.hpp>

namespace ai_stream {
namespace nodes {

class PoseInferNode : public IInferNode {
public:
    PoseInferNode();
    ~PoseInferNode() override;

    bool loadModel(const std::string& model_path) override;
    void setPrecision(const std::string& precision) override;
    void setBatchSize(int batch_size) override;
    std::pair<int, int> getInputSize() const override;
    std::vector<std::string> getClassNames() const override;

    void setDetectorType(DetectorType type) override { detector_type_ = type; }
    DetectorType getDetectorType() const override { return detector_type_; }

    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void inferLoop();
    bool initEngine(const std::string& engine_path);
    
    std::vector<std::shared_ptr<core::InferenceResultPacket>> processBatch(
        const std::vector<std::shared_ptr<core::VideoFramePacket>>& frames);

    void preprocessBatch(const std::vector<cv::Mat*>& images, float* gpu_buffer, int batch_size);
    
    // 后处理：每帧取最优候选，解码关键点
    void postprocessBatch(
        int batch_size,
        const float* host_output,  // [B, N, 56]
        const std::vector<cv::Rect2f>& crop_rois,
        const std::vector<cv::Size>& source_sizes,
        float conf_thresh,
        std::vector<std::shared_ptr<core::InferenceResultPacket>>& results);

    // TensorRT 资源
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_{
        nullptr, [](nvinfer1::IRuntime* p){ if (p) delete p; }};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_{
        nullptr, [](nvinfer1::ICudaEngine* p){ if (p) delete p; }};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_{
        nullptr, [](nvinfer1::IExecutionContext* p){ if (p) delete p; }};
    cudaStream_t stream_ = nullptr;

    // Tensor 名称（根据你的转置模型输出名修改）
    std::string input_name_ = "images";
    std::string output_name_ = "output0_transposed";  

    // GPU 缓冲区
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;

    // CPU 缓冲区
    std::vector<float> h_output_;

    // 常量
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;
    static constexpr int NUM_KPTS = 17;
    static constexpr int KPT_DIMS = 3;  // x, y, confidence
    static constexpr int OUT_DIM = 56;  // 4 box + 1 score + 51 kpts
    static constexpr int MAX_CANDIDATES = 8400;

    size_t input_size_ = 0;
    size_t output_size_ = 0;

    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 1;
    std::string precision_ = "fp16";
    DetectorType detector_type_ = DetectorType::POSE;

    std::vector<std::string> class_names_ = {"person"};

    // 配置
    float conf_thresh_ = 0.25f;
    float kpt_conf_thresh_ = 0.5f;

    // 数据队列和线程
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> queue_{64};
    std::thread worker_;
    std::atomic<bool> running_{false};

    std::chrono::milliseconds batch_timeout_ms_{20};
    std::atomic<int> max_batch_size_{1};
};

}}