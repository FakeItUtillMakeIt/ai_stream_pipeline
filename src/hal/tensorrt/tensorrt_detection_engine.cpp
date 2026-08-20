// src/hal/tensorrt/tensorrt_detection_engine.cpp
// TensorRT 检测推理引擎实现
#include "tensorrt_detection_engine.h"
#include "ai_stream/hal/detection_inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"
#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <fstream>
#include <iostream>

namespace ai_stream {
namespace hal {

TensorrtDetectionEngine::TensorrtDetectionEngine() {
    LOG_DEBUG("[TensorrtDetectionEngine] Constructor");
}

TensorrtDetectionEngine::~TensorrtDetectionEngine() {
    LOG_DEBUG("[TensorrtDetectionEngine] Destructor");
    freeBuffers();
}

void TensorrtDetectionEngine::deleteRuntime(nvinfer1::IRuntime* p) {
    delete p;
}

void TensorrtDetectionEngine::deleteEngine(nvinfer1::ICudaEngine* p) {
    delete p;
}

void TensorrtDetectionEngine::deleteContext(nvinfer1::IExecutionContext* p) {
    delete p;
}

bool TensorrtDetectionEngine::loadModel(const DetectionInferenceConfig& config) {
    LOG_INFO_FMT("[TensorrtDetectionEngine] Loading model from: {}", config.model_path);
    config_ = config;
    input_width_ = config.input_width;
    input_height_ = config.input_height;
    max_batch_size_ = config.max_batch_size;
    max_detections_ = config.max_detections;

    if (!initEngine(config.model_path)) {
        return false;
    }

    loaded_ = true;
    LOG_INFO_FMT("[TensorrtDetectionEngine] Model loaded: {} (max_batch={}, max_dets={})",
                 config.model_path, max_batch_size_, max_detections_);
    return true;
}

bool TensorrtDetectionEngine::setInputTensor(const std::string& name, void* gpu_ptr) {
    if (!context_) return false;
    tensor_ptrs_[name] = gpu_ptr;
    return context_->setTensorAddress(name.c_str(), gpu_ptr) == true;
}

bool TensorrtDetectionEngine::setOutputTensor(const std::string& name, void* gpu_ptr) {
    if (!context_) return false;
    tensor_ptrs_[name] = gpu_ptr;
    return context_->setTensorAddress(name.c_str(), gpu_ptr) == true;
}

void* TensorrtDetectionEngine::getOutputTensor(const std::string& name) {
    auto it = tensor_ptrs_.find(name);
    if (it != tensor_ptrs_.end()) {
        return it->second;
    }
    return nullptr;
}

size_t TensorrtDetectionEngine::getOutputTensorSize(const std::string& name) const {
    auto it = tensor_sizes_.find(name);
    if (it != tensor_sizes_.end()) {
        return it->second;
    }
    return 0;
}

bool TensorrtDetectionEngine::allocateOutputBuffers() {
    if (!engine_) return false;

    int max_batch = max_batch_size_;
    out_boxes_size_ = static_cast<size_t>(max_batch) * max_detections_ * 4 * sizeof(float);
    out_scores_size_ = static_cast<size_t>(max_batch) * max_detections_ * sizeof(float);
    out_classes_size_ = static_cast<size_t>(max_batch) * max_detections_ * sizeof(int64_t);
    out_batch_ids_size_ = static_cast<size_t>(max_batch) * max_detections_ * sizeof(int64_t);
    out_num_dets_size_ = sizeof(int64_t);

    cudaError_t err;
    err = cudaMalloc(&d_boxes_, out_boxes_size_);
    if (err != cudaSuccess) { LOG_ERROR("Failed to allocate d_boxes_"); return false; }
    err = cudaMalloc(&d_scores_, out_scores_size_);
    if (err != cudaSuccess) { LOG_ERROR("Failed to allocate d_scores_"); return false; }
    err = cudaMalloc(&d_classes_, out_classes_size_);
    if (err != cudaSuccess) { LOG_ERROR("Failed to allocate d_classes_"); return false; }
    err = cudaMalloc(&d_batch_ids_, out_batch_ids_size_);
    if (err != cudaSuccess) { LOG_ERROR("Failed to allocate d_batch_ids_"); return false; }
    err = cudaMalloc(&d_num_dets_, out_num_dets_size_);
    if (err != cudaSuccess) { LOG_ERROR("Failed to allocate d_num_dets_"); return false; }

    // 保存大小信息
    tensor_sizes_[boxes_name_] = out_boxes_size_;
    tensor_sizes_[scores_name_] = out_scores_size_;
    tensor_sizes_[classes_name_] = out_classes_size_;
    tensor_sizes_[batch_ids_name_] = out_batch_ids_size_;
    tensor_sizes_[num_dets_name_] = out_num_dets_size_;

    // 设置 tensor 地址
    setOutputTensor(boxes_name_, d_boxes_);
    setOutputTensor(scores_name_, d_scores_);
    setOutputTensor(classes_name_, d_classes_);
    setOutputTensor(batch_ids_name_, d_batch_ids_);
    setOutputTensor(num_dets_name_, d_num_dets_);

    LOG_INFO("[TensorrtDetectionEngine] Output buffers allocated");
    return true;
}

bool TensorrtDetectionEngine::infer() {
    if (!context_) return false;
    return context_->enqueueV3(0);
}

bool TensorrtDetectionEngine::inferAsync(void* stream) {
    if (!context_) return false;
    return context_->enqueueV3(static_cast<cudaStream_t>(stream));
}

bool TensorrtDetectionEngine::synchronize(void* stream) {
    if (!stream) return false;
    cudaError_t err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
    return err == cudaSuccess;
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
    return context_.get();
}

void* TensorrtDetectionEngine::getRawEngine() const {
    return engine_.get();
}

nvinfer1::IExecutionContext* TensorrtDetectionEngine::getTensorRTContext() const {
    return context_.get();
}

nvinfer1::ICudaEngine* TensorrtDetectionEngine::getTensorRTEngine() const {
    return engine_.get();
}

bool TensorrtDetectionEngine::initEngine(const std::string& engine_path) {
    try {
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR_FMT("[TensorrtDetectionEngine] Failed to open engine: {}", engine_path);
            return false;
        }

        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            LOG_ERROR("[TensorrtDetectionEngine] Failed to read engine");
            return false;
        }

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
        if (!runtime) {
            LOG_ERROR("[TensorrtDetectionEngine] createInferRuntime failed");
            return false;
        }

        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
        if (!engine) {
            LOG_ERROR("[TensorrtDetectionEngine] deserializeCudaEngine failed");
            delete runtime;
            return false;
        }

        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        if (!context) {
            LOG_ERROR("[TensorrtDetectionEngine] createExecutionContext failed");
            delete engine;
            delete runtime;
            return false;
        }

        runtime_.reset(runtime);
        engine_.reset(engine);
        context_.reset(context);

        // 打印 tensor 信息
        int nb_io = engine_->getNbIOTensors();
        LOG_INFO_FMT("[TensorrtDetectionEngine] Engine has {} I/O tensors", nb_io);
        for (int i = 0; i < nb_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
            LOG_INFO_FMT("[TensorrtDetectionEngine]  Tensor[{}]: {} ({}), shape=[{}]",
                         i, name, mode_str, dims.nbDims);
        }

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[TensorrtDetectionEngine] initEngine exception: {}", e.what());
        return false;
    }
}

void TensorrtDetectionEngine::freeBuffers() {
    if (d_boxes_) { cudaFree(d_boxes_); d_boxes_ = nullptr; }
    if (d_scores_) { cudaFree(d_scores_); d_scores_ = nullptr; }
    if (d_classes_) { cudaFree(d_classes_); d_classes_ = nullptr; }
    if (d_batch_ids_) { cudaFree(d_batch_ids_); d_batch_ids_ = nullptr; }
    if (d_num_dets_) { cudaFree(d_num_dets_); d_num_dets_ = nullptr; }
    tensor_ptrs_.clear();
    tensor_sizes_.clear();
}

// 注册到工厂
#ifdef WITH_CUDA
REGISTER_DETECTION_INFERENCE_BACKEND(DetectionBackend::TENSORRT, TensorrtDetectionEngine)
#endif

} // namespace hal
} // namespace ai_stream
