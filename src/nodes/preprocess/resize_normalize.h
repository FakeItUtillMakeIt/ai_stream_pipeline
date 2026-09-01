// src/nodes/preprocess/resize_normalize.h
#pragma once

#include "ai_stream/nodes/i_preprocess_node.h"
#include "ai_stream/core/queued_node.h"
#include "ai_stream/hal/i_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include <opencv2/core/mat.hpp>

namespace ai_stream {
namespace nodes {

class ResizeNormalizeNode : public core::QueuedNode<IPreprocessNode> {
public:
    ResizeNormalizeNode();
    ~ResizeNormalizeNode();

    bool acceptsGpuFrame() const override { return true; }

    // QueuedNode 接口
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;
    bool onStartup() override;
    void onShutdown() override;

    // 配置方法（可通过 JSON params 设置）
    void setTargetSize(int width, int height);
    std::pair<int, int> getTargetSize() const override;
    void setMean(const std::vector<float>& mean);
    void setStd(const std::vector<float>& std);
    void setInterpolationMethod(const std::string& method);
    void setKeepAspectRatio(bool keep_aspect_ratio);
    void setOutputDataType(const std::string& dtype);
    void setImageAcceleratorBackend(hal::ImageAcceleratorBackend backend);

private:
    bool processCpuFrame(const std::shared_ptr<core::VideoFramePacket>& frame);

#ifdef WITH_CUDA
    bool processGpuFrame(const std::shared_ptr<core::VideoFramePacket>& frame);
#endif

    hal::IImageAccelerator* getCpuAccelerator();
#ifdef WITH_CUDA
    hal::IImageAccelerator* getGpuAccelerator();
#endif

    std::shared_ptr<cv::Mat> convertNchwToMat(const float* nchw, int width, int height) const;

    hal::ImageAcceleratorBackend backend_type_ = hal::ImageAcceleratorBackend::AUTO;
    hal::ImageAcceleratorBackend cpu_backend_selected_ = hal::ImageAcceleratorBackend::AUTO;
#ifdef WITH_CUDA
    hal::ImageAcceleratorBackend gpu_backend_selected_ = hal::ImageAcceleratorBackend::AUTO;
    bool gpu_backend_checked_ = false;
#endif

    hal::ImageAcceleratorPtr cpu_accelerator_;
#ifdef WITH_CUDA
    hal::ImageAcceleratorPtr gpu_accelerator_;
    // GPU 输入（BGR 上传）与输出（NCHW）缓冲区由 GpuBufferPool 按帧分配，
    // 所有权随 packet 传递（d_buf_owner / d_bgr_owner），下游队列再深也不会
    // 被后续帧覆盖。
    void* cuda_stream_ = nullptr;
    bool owns_cuda_stream_ = false;
#endif

    std::vector<float> host_nchw_buffer_;

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
