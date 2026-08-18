// src/nodes/infer/cuda_pose_infer.h
#pragma once

#include "ai_stream/nodes/i_infer_node.h"
#include "ai_stream/core/bounded_queue.h"

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

class CudaPoseInferNode : public IInferNode {
public:
    CudaPoseInferNode();
    ~CudaPoseInferNode() override;

    bool loadModel(const std::string& model_path) override;
    void setPrecision(const std::string& precision) override;
    void setBatchSize(int batch_size) override;
    void setInputSize(int width, int height) override { input_width_ = width; input_height_ = height;}
    std::pair<int, int> getInputSize() const override;
    void setClassNames(const std::vector<std::string>& names) override { class_names_ = names; }
    std::vector<std::string> getClassNames() const override;

    void setDetectorType(DetectorType type) override { detector_type_ = type; }
    DetectorType getDetectorType() const override { return detector_type_; }

    bool start() override;
    void stop() override;
    bool isRunning() const override{return running_.load();}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void inferLoop();
    bool initEngine(const std::string& engine_path);

    // 处理单帧 InferenceResultPacket
    void processFrame(std::shared_ptr<core::InferenceResultPacket> packet);

    // GPU 预处理：确保原图上传缓冲区足够
    void ensureSourceBuffer(int w, int h);

    // 单帧内多人 batch 后处理
    void postprocessFrame(
        std::shared_ptr<core::InferenceResultPacket> packet,
        const std::vector<int>& person_indices,
        int num_persons,
        float* output_host,
        const std::vector<float>& letterbox_params);

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
    std::string output_name_ = "output0_transposed";  // 转置后的输出

    // GPU 缓冲区
    void* d_input_ = nullptr;
    void* d_output_ = nullptr;
    void* d_source_img_ = nullptr;   // 原图 GPU 缓冲区 (uint8 BGR)
    float* d_boxes_ = nullptr;       // 检测框 GPU 缓冲区 [batch, 4]

    // CPU 缓冲区
    std::vector<float> h_output_;  // [max_batch, 8400, 56]
    std::vector<float> h_letterbox_params_; // [num_persons, 3] (scale, pad_x, pad_y)

    // 缓冲区大小跟踪
    size_t source_buffer_size_ = 0;
    size_t input_size_ = 0;
    size_t output_size_ = 0;

    // 常量
    static constexpr int INPUT_H = 640;
    static constexpr int INPUT_W = 640;
    static constexpr int NUM_CANDIDATES = 8400;  // 80*80 + 40*40 + 20*20
    static constexpr int POSE_DIM = 56;           // 4 box + 1 score + 51 kpts
    static constexpr int NUM_KEYPOINTS = 17;

    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 8;   // 单帧最大人数
    std::string precision_ = "fp16";
    DetectorType detector_type_ = DetectorType::POSE;

    // 过滤参数
    float conf_thresh_ = 0.25f;      // person 置信度阈值
    float kpt_conf_thresh_ = 0.5f;   // 关键点可见度阈值
    int person_class_id_ = 0;        // person 的 class_id

    std::vector<std::string> class_names_ = {
        "person"
    };

    // 数据队列和线程
    core::BoundedQueue<std::shared_ptr<core::InferenceResultPacket>> queue_{64};
    std::thread worker_;
    std::atomic<bool> running_{false};

    // 耗时统计
    uint64_t in_time_ms_ = 0;
};

} // namespace nodes
} // namespace ai_stream