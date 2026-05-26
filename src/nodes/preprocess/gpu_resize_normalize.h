// src/nodes/preprocess/gpu_resize_normalize.h
#pragma once

#include "ai_stream/nodes/i_gpu_preprocess_node.h"
#include <cuda_runtime.h>

namespace ai_stream {
namespace nodes {

/**
 * @brief GPU 加速的预处理节点实现
 *
 * 使用 CUDA 和 NPP 库实现高性能的图像预处理。
 * 支持 GPU 加速的缩放、颜色空间转换和归一化操作。
 */
class GPUResizeNormalizeNode : public IGpuPreprocessNode {
public:
    GPUResizeNormalizeNode();
    ~GPUResizeNormalizeNode() override;

    // Node 接口
    bool start() override;
    void stop() override;
    bool isRunning() const override{return running_.load();}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    // IPreprocessNode 接口
    void setTargetSize(int width, int height) { target_width_ = width; target_height_ = height; }
    std::pair<int, int> getTargetSize() const { return std::make_pair(target_width_, target_height_); }
    void setMean(const std::vector<float>& mean) { mean_ = mean; }
    void setStd(const std::vector<float>& std) { std_ = std; }
    void setInterpolationMethod(const std::string& method) { interpolation_method_ = method; }
    void setKeepAspectRatio(bool keep_aspect_ratio) { keep_aspect_ratio_ = keep_aspect_ratio; }
    void setOutputDataType(const std::string& dtype) { output_dtype_ = dtype; }

    // IGpuPreprocessNode 接口
    void setGpuDeviceId(int device_id) override;
    int getGpuDeviceId() const override;
    void setAsyncProcessing(bool async) override;
    void setCudaStream(cudaStream_t stream) override;
    float getAverageLatencyMs() const override;
    void setTensorRTPreprocessEnabled(bool enable) override;

private:
    int device_id_;
    bool async_processing_;
    cudaStream_t stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

    // GPU 内存缓冲区
    void* input_gpu_ptr_;
    void* output_gpu_ptr_;
    size_t gpu_buffer_size_;

    // 预处理参数
    int target_width_;
    int target_height_;
    std::vector<float> mean_;
    std::vector<float> std_;
    std::string interpolation_method_;
    bool keep_aspect_ratio_;
    std::string output_dtype_;

    // 性能统计
    std::atomic<int> total_processed_;
    std::atomic<int64_t> total_latency_ms_;
    std::atomic<bool> tensorrt_preprocess_enabled_;
};

} // namespace nodes
} // namespace ai_stream