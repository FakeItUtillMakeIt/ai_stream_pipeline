// src/nodes/preprocess/cuda_resize_normalize.h
#pragma once

#include "ai_stream/nodes/i_gpu_preprocess_node.h"
#include <cuda_runtime.h>

namespace ai_stream {
namespace nodes {

class CudaResizeNormalizeNode : public IGpuPreprocessNode {
public:
    CudaResizeNormalizeNode();
    ~CudaResizeNormalizeNode() override;

    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    void setTargetSize(int width, int height) { target_width_ = width; target_height_ = height; }
    std::pair<int, int> getTargetSize() const { return std::make_pair(target_width_, target_height_); }
    void setMean(const std::vector<float>& mean);
    void setStd(const std::vector<float>& std);
    void setInterpolationMethod(const std::string& method) { interpolation_method_ = method; }
    void setKeepAspectRatio(bool keep_aspect_ratio) { keep_aspect_ratio_ = keep_aspect_ratio; }
    void setOutputDataType(const std::string& dtype) { output_dtype_ = dtype; }

    void setGpuDeviceId(int device_id) override;
    int getGpuDeviceId() const override;
    void setAsyncProcessing(bool async) override;
    void setCudaStream(cudaStream_t stream) override;
    float getAverageLatencyMs() const override;
    void setTensorRTPreprocessEnabled(bool enable) override;

private:
    // 按需分配/扩容 GPU 缓冲区
    void ensureGpuBuffers(int src_w, int src_h, int dst_w, int dst_h);

    int device_id_;
    bool async_processing_;
    cudaStream_t stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

    void* input_gpu_ptr_;
    void* output_gpu_ptr_;
    size_t input_buffer_size_;   // 当前输入缓冲区字节数
    size_t output_buffer_size_;  // 当前输出缓冲区字节数

    int target_width_=640;
    int target_height_=640;
    std::vector<float> mean_;
    std::vector<float> std_;
    std::string interpolation_method_;
    bool keep_aspect_ratio_;
    std::string output_dtype_;

    std::atomic<int> total_processed_;
    std::atomic<int64_t> total_latency_ms_;
    std::atomic<bool> tensorrt_preprocess_enabled_;
    std::atomic<bool> running_{false};
};

} // namespace nodes
} // namespace ai_stream