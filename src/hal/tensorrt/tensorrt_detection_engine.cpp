// src/hal/tensorrt/tensorrt_detection_engine.cpp
// TensorRT 检测推理引擎实现
#include "tensorrt_detection_engine.h"
#include "ai_stream/hal/detection_inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace ai_stream {
namespace hal {

TensorrtDetectionEngine::TensorrtDetectionEngine() {
    LOG_DEBUG("[TensorrtDetectionEngine] Constructor");
}

TensorrtDetectionEngine::~TensorrtDetectionEngine() {
    LOG_DEBUG("[TensorrtDetectionEngine] Destructor");
}

bool TensorrtDetectionEngine::loadModel(const DetectionInferenceConfig& config) {
    LOG_INFO_FMT("[TensorrtDetectionEngine] Loading model from: {}", config.model_path);
    config_ = config;
    input_width_ = config.input_width;
    input_height_ = config.input_height;
    max_batch_size_ = config.max_batch_size;
    max_detections_ = config.max_detections;

    if (!core_.loadEngine(config.model_path, "TensorrtDetectionEngine")) {
        return false;
    }

    // 确定输入名（第一个输入）
    for (const auto& meta : core_.tensors()) {
        if (meta.is_input) {
            input_name_ = meta.name;
            break;
        }
    }

    loaded_ = true;
    LOG_INFO_FMT("[TensorrtDetectionEngine] Model loaded: {} (max_batch={}, max_dets={})",
                 config.model_path, max_batch_size_, max_detections_);
    return true;
}

bool TensorrtDetectionEngine::setInputTensor(const std::string& name, void* gpu_ptr) {
    tensor_ptrs_[name] = gpu_ptr;
    return core_.setAddress(name, gpu_ptr);
}

bool TensorrtDetectionEngine::setOutputTensor(const std::string& name, void* gpu_ptr) {
    tensor_ptrs_[name] = gpu_ptr;
    return core_.setAddress(name, gpu_ptr);
}

void* TensorrtDetectionEngine::getOutputTensor(const std::string& name) {
    // 优先返回节点自管缓冲区，其次引擎内部分配的
    auto it = tensor_ptrs_.find(name);
    if (it != tensor_ptrs_.end()) {
        return it->second;
    }
    return core_.buffer(name);
}

size_t TensorrtDetectionEngine::getOutputTensorSize(const std::string& name) const {
    return core_.bufferSize(name);
}

bool TensorrtDetectionEngine::allocateOutputBuffers() {
    if (!core_.isLoaded()) return false;

    int max_batch = max_batch_size_;
    const size_t boxes_bytes =
        static_cast<size_t>(max_batch) * max_detections_ * 4 * sizeof(float);
    const size_t scores_bytes =
        static_cast<size_t>(max_batch) * max_detections_ * sizeof(float);
    const size_t classes_bytes =
        static_cast<size_t>(max_batch) * max_detections_ * sizeof(int64_t);
    const size_t batch_ids_bytes = classes_bytes;
    const size_t num_dets_bytes = sizeof(int64_t);

    if (!core_.allocBuffer(boxes_name_, boxes_bytes)) return false;
    if (!core_.allocBuffer(scores_name_, scores_bytes)) return false;
    if (!core_.allocBuffer(classes_name_, classes_bytes)) return false;
    if (!core_.allocBuffer(batch_ids_name_, batch_ids_bytes)) return false;
    if (!core_.allocBuffer(num_dets_name_, num_dets_bytes)) return false;

    LOG_INFO("[TensorrtDetectionEngine] Output buffers allocated");
    return true;
}

bool TensorrtDetectionEngine::infer() {
    return core_.enqueue();
}

bool TensorrtDetectionEngine::inferAsync(void* stream) {
    // 临时切换流执行
    void* prev = core_.stream();
    core_.setStream(stream);
    bool ok = core_.enqueue();
    core_.setStream(prev);
    return ok;
}

bool TensorrtDetectionEngine::synchronize(void* stream) {
    if (!stream) return false;
    return cudaStreamSynchronize(static_cast<cudaStream_t>(stream)) == cudaSuccess;
}

std::vector<std::string> TensorrtDetectionEngine::getInputNames() const {
    return {input_name_};
}

std::vector<std::string> TensorrtDetectionEngine::getOutputNames() const {
    return {boxes_name_, scores_name_, classes_name_, batch_ids_name_, num_dets_name_};
}

std::pair<int, int> TensorrtDetectionEngine::getInputSize() const {
    return {input_width_, input_height_};
}

int TensorrtDetectionEngine::getMaxBatchSize() const {
    return max_batch_size_;
}

bool TensorrtDetectionEngine::isAvailable() const {
#ifdef WITH_CUDA
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return err == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

void* TensorrtDetectionEngine::getRawContext() const {
    return core_.context();
}

void* TensorrtDetectionEngine::getRawEngine() const {
    return core_.engine();
}

nvinfer1::IExecutionContext* TensorrtDetectionEngine::getTensorRTContext() const {
    return static_cast<nvinfer1::IExecutionContext*>(core_.context());
}

nvinfer1::ICudaEngine* TensorrtDetectionEngine::getTensorRTEngine() const {
    return static_cast<nvinfer1::ICudaEngine*>(core_.engine());
}

// 注册到工厂
#ifdef WITH_CUDA
REGISTER_DETECTION_INFERENCE_BACKEND(DetectionBackend::TENSORRT, TensorrtDetectionEngine)
#endif

} // namespace hal
} // namespace ai_stream
