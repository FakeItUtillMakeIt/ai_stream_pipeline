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
#include <deque>

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
    bool isRunning() const override { return running_.load(); }
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void inferLoop();
    bool initEngine(const std::string& engine_path);
    
    std::vector<std::shared_ptr<core::InferenceResultPacket>> processBatch(
        const std::vector<std::shared_ptr<core::VideoFramePacket>>& frames);

    // 【新增】双路径预处理
    void preprocessBatchCpu(const std::vector<cv::Mat*>& images, float* gpu_buffer, int batch_size);
    void preprocessBatchGpu(const std::vector<void*>& d_ptrs, const std::vector<size_t>& pitches,
                            float* gpu_buffer, int batch_size);

    std::vector<std::vector<core::InferenceResultPacket::BBox>> postprocessBatch(
        int batch_size, int total_dets, const float scale_x[], const float scale_y[], float conf_thresh);

    // TensorRT 资源
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_{
        nullptr, [](nvinfer1::IRuntime* p){ if (p) delete p; }};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_{
        nullptr, [](nvinfer1::ICudaEngine* p){ if (p) delete p; }};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_{
        nullptr, [](nvinfer1::IExecutionContext* p){ if (p) delete p; }};
    cudaStream_t stream_ = nullptr;

    // Tensor 名称
    std::string input_name_ = "images";
    std::string boxes_name_ = "det_boxes";
    std::string scores_name_ = "det_scores";
    std::string classes_name_ = "det_classes";
    std::string batch_ids_name_ = "det_batch_ids";
    std::string num_dets_name_ = "det_num_dets";

    // GPU 缓冲区
    void* d_input_ = nullptr;
    void* d_boxes_ = nullptr;
    void* d_scores_ = nullptr;
    void* d_classes_ = nullptr;
    void* d_batch_ids_ = nullptr;
    void* d_num_dets_ = nullptr;

    // 【新增】用于 GPU 路径的临时缓冲区（当需要数据整理时）
    void* d_preprocess_tmp_ = nullptr;
    size_t d_preprocess_tmp_size_ = 0;

    // CPU 缓冲区
    std::vector<float> h_boxes_;
    std::vector<float> h_scores_;
    std::vector<int64_t> h_classes_;
    std::vector<int64_t> h_batch_ids_;
    int64_t h_num_dets_ = 0;

    // 常量
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;
    static constexpr int MAX_DETS = 20;
    
    size_t input_size_ = 0;
    size_t out_boxes_size_ = 0;
    size_t out_scores_size_ = 0;
    size_t out_classes_size_ = 0;
    size_t out_batch_ids_size_ = 0;
    size_t out_num_dets_size_ = 0;

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

    // 数据队列和线程
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> queue_{64};
    std::thread worker_;
    std::atomic<bool> running_{false};

    std::chrono::milliseconds batch_timeout_ms_{20};
    std::atomic<int> max_batch_size_{1};
};

} // namespace nodes
} // namespace ai_stream