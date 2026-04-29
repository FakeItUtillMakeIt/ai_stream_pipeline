// src/nodes/infer/detection_infer.h
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

namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace ai_stream {
namespace nodes {

class DetectionInferNode : public IInferNode {
public:
    DetectionInferNode();
    ~DetectionInferNode() override;

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
    
    std::shared_ptr<core::InferenceResultPacket> processFrame(
        std::shared_ptr<core::VideoFramePacket> frame);

    void preprocess(const cv::Mat& image, float* gpu_buffer);
    
    std::vector<core::InferenceResultPacket::BBox> postprocess(
        int num_dets, float scale_x, float scale_y, float conf_thresh);

    // TensorRT 资源（TensorRT 10 API）
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_{
        nullptr, [](nvinfer1::IRuntime* p){ if (p) delete p; }};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_{
        nullptr, [](nvinfer1::ICudaEngine* p){ if (p) delete p; }};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_{
        nullptr, [](nvinfer1::IExecutionContext* p){ if (p) delete p; }};
    cudaStream_t stream_ = nullptr;

    // Tensor 名称（TensorRT 10 用名称而非索引）
    std::string input_name_ = "images";
    std::string boxes_name_ = "det_boxes";
    std::string scores_name_ = "det_scores";
    std::string classes_name_ = "det_classes";

    // GPU 缓冲区
    void* d_input_ = nullptr;
    void* d_boxes_ = nullptr;
    void* d_scores_ = nullptr;
    void* d_classes_ = nullptr;

    // CPU 缓冲区
    std::vector<float> h_boxes_;
    std::vector<float> h_scores_;
    std::vector<int64_t> h_classes_;

    // 常量
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;
    static constexpr int MAX_DETS = 300;
    
    size_t input_size_ = 0;
    size_t out_boxes_size_ = 0;
    size_t out_scores_size_ = 0;
    size_t out_classes_size_ = 0;

    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 1;
    std::string precision_ = "fp16";
    DetectorType detector_type_ = DetectorType::DETECTION;

    std::vector<std::string> class_names_ = {
        "person", "head", "helmet", "clothes_red", "clothes_gray", 
        "clothes_yellow", "clothes_blue", "clothes_similar",
        "clothes_reflective", "phone", "smoking", "fall", 
        "safety_belt", "sleeping"
    };

    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> queue_{4};
    std::thread worker_;
    std::atomic<bool> running_{false};
};

} // namespace nodes
} // namespace ai_stream