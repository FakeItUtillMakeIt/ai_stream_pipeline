// src/hal/tensorrt/tensorrt_detection_engine.h
// TensorRT 检测推理引擎——封装现有 TensorRT 逻辑到 HAL 接口
#pragma once

#include "ai_stream/hal/i_detection_inference_engine.h"
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>

// 前向声明 TensorRT 类型
namespace nvinfer1 {
    class IRuntime;
    class ICudaEngine;
    class IExecutionContext;
}

namespace ai_stream {
namespace hal {

/**
 * @brief TensorRT 检测推理引擎
 *
 * 封装 TensorRT 推理逻辑到 IDetectionInferenceEngine 接口。
 * 支持多输出 tensor、异步推理、CUDA Graph。
 */
class TensorrtDetectionEngine : public IDetectionInferenceEngine {
public:
    TensorrtDetectionEngine();
    ~TensorrtDetectionEngine() override;

    bool loadModel(const DetectionInferenceConfig& config) override;
    bool setInputTensor(const std::string& name, void* gpu_ptr) override;
    bool setOutputTensor(const std::string& name, void* gpu_ptr) override;
    void* getOutputTensor(const std::string& name) override;
    size_t getOutputTensorSize(const std::string& name) const override;
    bool allocateOutputBuffers() override;
    bool infer() override;
    bool inferAsync(void* stream) override;
    bool synchronize(void* stream) override;
    std::vector<std::string> getInputNames() const override;
    std::vector<std::string> getOutputNames() const override;
    std::pair<int, int> getInputSize() const override;
    int getMaxBatchSize() const override;
    std::string getBackendName() const override { return "TensorRT (NVIDIA)"; }
    bool isAvailable() const override;
    void* getRawContext() const override;
    void* getRawEngine() const override;

    /**
     * @brief 获取原始 TensorRT context（用于 CUDA Graph 等高级优化）
     */
    nvinfer1::IExecutionContext* getTensorRTContext() const;

    /**
     * @brief 获取原始 TensorRT engine
     */
    nvinfer1::ICudaEngine* getTensorRTEngine() const;

private:
    static void deleteRuntime(nvinfer1::IRuntime* p);
    static void deleteEngine(nvinfer1::ICudaEngine* p);
    static void deleteContext(nvinfer1::IExecutionContext* p);

    bool initEngine(const std::string& engine_path);
    bool allocateBuffers();
    void freeBuffers();

    DetectionInferenceConfig config_;
    bool loaded_ = false;

    // TensorRT 资源
    std::unique_ptr<nvinfer1::IRuntime, void(*)(nvinfer1::IRuntime*)> runtime_{
        nullptr, &TensorrtDetectionEngine::deleteRuntime};
    std::unique_ptr<nvinfer1::ICudaEngine, void(*)(nvinfer1::ICudaEngine*)> engine_{
        nullptr, &TensorrtDetectionEngine::deleteEngine};
    std::unique_ptr<nvinfer1::IExecutionContext, void(*)(nvinfer1::IExecutionContext*)> context_{
        nullptr, &TensorrtDetectionEngine::deleteContext};

    // Tensor 名称
    std::string input_name_ = "images";
    std::string boxes_name_ = "det_boxes";
    std::string scores_name_ = "det_scores";
    std::string classes_name_ = "det_classes";
    std::string batch_ids_name_ = "det_batch_ids";
    std::string num_dets_name_ = "det_num_dets";

    // GPU 缓冲区（由节点管理，引擎只保存指针）
    std::unordered_map<std::string, void*> tensor_ptrs_;
    std::unordered_map<std::string, size_t> tensor_sizes_;

    // 输出缓冲区（可选，由引擎管理时使用）
    void* d_boxes_ = nullptr;
    void* d_scores_ = nullptr;
    void* d_classes_ = nullptr;
    void* d_batch_ids_ = nullptr;
    void* d_num_dets_ = nullptr;

    // 输入尺寸
    int input_width_ = 640;
    int input_height_ = 640;
    int max_batch_size_ = 1;
    int max_detections_ = 200;

    // 计算的大小
    size_t out_boxes_size_ = 0;
    size_t out_scores_size_ = 0;
    size_t out_classes_size_ = 0;
    size_t out_batch_ids_size_ = 0;
    size_t out_num_dets_size_ = 0;
};

} // namespace hal
} // namespace ai_stream
