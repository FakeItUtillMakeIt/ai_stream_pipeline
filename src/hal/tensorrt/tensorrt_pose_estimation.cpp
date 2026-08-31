// src/hal/tensorrt/tensorrt_pose_estimation.cpp
#include "tensorrt_pose_estimation.h"
#include "tensorrt_pose_kernels.cuh"
#include "ai_stream/hal/pose_estimation_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"

#include <algorithm>
#include <NvInfer.h>
#include <cuda_runtime.h>

namespace ai_stream {
namespace hal {

TensorrtPoseEstimation::TensorrtPoseEstimation() = default;

TensorrtPoseEstimation::~TensorrtPoseEstimation() = default;

bool TensorrtPoseEstimation::isAvailable() const {
#ifdef WITH_TENSORRT
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

void TensorrtPoseEstimation::setCudaStream(void* stream) {
    core_.setStream(stream);
}

size_t TensorrtPoseEstimation::getOutputFloatsPerPerson() const {
    return static_cast<size_t>(NUM_CANDIDATES) * POSE_DIM;
}

std::pair<int, int> TensorrtPoseEstimation::getInputSize() const {
    return {input_w_, input_h_};
}

bool TensorrtPoseEstimation::loadModel(const PoseEstimationConfig& config) {
    config_ = config;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        config.device_id < 0 || config.device_id >= device_count ||
        cudaSetDevice(config.device_id) != cudaSuccess) {
        LOG_ERROR_FMT("[TensorrtPose] Cannot select CUDA device {}", config.device_id);
        return false;
    }

    if (!core_.loadEngine(config.model_path, "TensorrtPose")) {
        return false;
    }

    // 确定 IO tensor 名称与实际输入尺寸（动态维度用配置值）
    bool output_found = false;
    for (const auto& meta : core_.tensors()) {
        if (meta.is_input) {
            input_name_ = meta.name;
            if (meta.nb_dims >= 4) {
                input_h_ = (meta.d[2] > 0) ? static_cast<int>(meta.d[2]) : config.input_height;
                input_w_ = (meta.d[3] > 0) ? static_cast<int>(meta.d[3]) : config.input_width;
            }
        } else if (!output_found) {
            output_name_ = meta.name;
            output_found = true;
        }
    }

    // 按最大 batch 预分配缓冲区
    const size_t input_capacity = static_cast<size_t>(config_.max_batch) * 3 *
                                  input_h_ * input_w_ * sizeof(float);
    const size_t output_capacity = static_cast<size_t>(config_.max_batch) *
                                   NUM_CANDIDATES * POSE_DIM * sizeof(float);
    if (!core_.allocBuffer(input_name_, input_capacity)) {
        return false;
    }
    if (!core_.allocBuffer(output_name_, output_capacity)) {
        return false;
    }
    if (!core_.allocBuffer("__boxes__",
                           static_cast<size_t>(config_.max_batch) * 7 * sizeof(float))) {
        return false;
    }

    loaded_ = true;
    LOG_INFO_FMT("[TensorrtPose] Engine loaded: {} (max_batch={}, input={}x{}, candidates={})",
                 config.model_path, config_.max_batch, input_w_, input_h_, NUM_CANDIDATES);
    return true;
}

bool TensorrtPoseEstimation::inferHost(const float* input_nchw, int num_persons,
                                       std::vector<float>& output_host) {
    if (!loaded_) {
        LOG_ERROR("[TensorrtPose] Model not loaded");
        return false;
    }
    if (!input_nchw || num_persons <= 0 || num_persons > config_.max_batch) return false;

    size_t bytes = static_cast<size_t>(num_persons) * 3 * input_h_ * input_w_ * sizeof(float);
    void* d_input = core_.buffer(input_name_);
    cudaError_t err = cudaMemcpyAsync(d_input, input_nchw, bytes,
                                      cudaMemcpyHostToDevice,
                                      static_cast<cudaStream_t>(core_.stream()));
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("[TensorrtPose] H2D copy failed: {}", cudaGetErrorString(err));
        return false;
    }
    return enqueueAndFetch(num_persons, d_input, output_host);
}

bool TensorrtPoseEstimation::inferDevice(const void* d_input_nchw, int num_persons,
                                         std::vector<float>& output_host) {
    if (!loaded_) {
        LOG_ERROR("[TensorrtPose] Model not loaded");
        return false;
    }
    if (!d_input_nchw || num_persons <= 0 || num_persons > config_.max_batch) return false;

    return enqueueAndFetch(num_persons, d_input_nchw, output_host);
}

bool TensorrtPoseEstimation::inferFromDeviceImage(
    const void* d_src_bgr, int src_w, int src_h, size_t src_pitch,
    const std::vector<float>& boxes_7, int num_persons,
    std::vector<float>& output_host)
{
    if (!loaded_) {
        LOG_ERROR("[TensorrtPose] Model not loaded");
        return false;
    }
    if (!d_src_bgr || num_persons <= 0 ||
        num_persons > config_.max_batch ||
        boxes_7.size() < static_cast<size_t>(num_persons) * 7) {
        return false;
    }

    auto stream = static_cast<cudaStream_t>(core_.stream());
    float* d_boxes = static_cast<float*>(core_.buffer("__boxes__"));

    // 上传检测框参数（独立缓冲区，避免与图像输出冲突）
    cudaError_t err = cudaMemcpyAsync(d_boxes, boxes_7.data(),
                                      num_persons * 7 * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("[TensorrtPose] boxes H2D failed: {}", cudaGetErrorString(err));
        return false;
    }

#ifdef WITH_CUDA
    // 启动融合预处理 kernel 直接生成 NCHW 输入
    launchCropResizeNormalizeLetterbox(
        static_cast<const unsigned char*>(d_src_bgr),
        src_w, src_h, src_pitch,
        static_cast<float*>(core_.buffer(input_name_)),
        d_boxes,
        num_persons,
        input_w_, input_h_,
        0.0f, 0.0f, 0.0f,
        1.0f, 1.0f, 1.0f,
        114.0f / 255.0f,
        stream);
#else
    LOG_ERROR("[TensorrtPose] inferFromDeviceImage requires WITH_CUDA build");
    return false;
#endif

    return enqueueAndFetch(num_persons, core_.buffer(input_name_), output_host);
}

bool TensorrtPoseEstimation::enqueueAndFetch(int num_persons, const void* d_input,
                                             std::vector<float>& output_host) {
    int64_t dims[4] = {num_persons, 3, input_h_, input_w_};
    if (!core_.setInputShape(input_name_, dims, 4)) {
        LOG_ERROR_FMT("[TensorrtPose] setInputShape failed for batch={}", num_persons);
        return false;
    }

    if (!core_.setAddress(input_name_, const_cast<void*>(d_input))) {
        return false;
    }

    if (!core_.enqueue()) {
        LOG_ERROR_FMT("[TensorrtPose] enqueueV3 failed for batch={}", num_persons);
        return false;
    }

    size_t output_bytes = static_cast<size_t>(num_persons) * NUM_CANDIDATES * POSE_DIM * sizeof(float);
    output_host.resize(static_cast<size_t>(num_persons) * NUM_CANDIDATES * POSE_DIM);
    cudaError_t err = cudaMemcpyAsync(output_host.data(), core_.buffer(output_name_),
                                      output_bytes, cudaMemcpyDeviceToHost,
                                      static_cast<cudaStream_t>(core_.stream()));
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("[TensorrtPose] D2H copy failed: {}", cudaGetErrorString(err));
        return false;
    }
    core_.synchronize();
    return true;
}

// 注册 TensorRT 姿态估计后端到工厂
#ifdef WITH_TENSORRT
REGISTER_POSE_ESTIMATION_BACKEND(PoseEstimationBackend::TENSORRT, TensorrtPoseEstimation)
#endif

} // namespace hal
} // namespace ai_stream
