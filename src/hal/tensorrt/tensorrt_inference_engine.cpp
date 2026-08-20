// src/hal/tensorrt/tensorrt_inference_engine.cpp
// TensorRT 推理引擎后端——封装现有 TensorRT 逻辑到 HAL 接口
#include "tensorrt_inference_engine.h"
#include "ai_stream/hal/inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"

#include <fstream>
#include <iostream>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace ai_stream {
namespace hal {

TensorrtInferenceEngine::TensorrtInferenceEngine() {
    LOG_DEBUG("[TensorrtInferenceEngine] Constructor");
}

TensorrtInferenceEngine::~TensorrtInferenceEngine() {
    freeBuffers();
    LOG_DEBUG("[TensorrtInferenceEngine] Destructor");
}

void TensorrtInferenceEngine::deleteRuntime(nvinfer1::IRuntime* p) {
    delete p;
}

void TensorrtInferenceEngine::deleteEngine(nvinfer1::ICudaEngine* p) {
    delete p;
}

void TensorrtInferenceEngine::deleteContext(nvinfer1::IExecutionContext* p) {
    delete p;
}

bool TensorrtInferenceEngine::loadModel(const InferenceConfig& config) {
    config_ = config;
    return initEngine(config.model_path) && allocateBuffers();
}

bool TensorrtInferenceEngine::infer(const void* input_data, size_t input_size,
                                     void* output_data, size_t output_size) {
    if (!loaded_) {
        LOG_ERROR("[TensorrtInferenceEngine] Model not loaded");
        return false;
    }

    // 拷贝输入到 GPU
    if (input_size > input_size_) {
        input_size = input_size_;
    }
    cudaMemcpyAsync(d_input_, input_data, input_size, cudaMemcpyHostToDevice,
                    static_cast<cudaStream_t>(cuda_stream_));

    // 执行推理
    if (!context_->enqueueV3(static_cast<cudaStream_t>(cuda_stream_))) {
        LOG_ERROR("[TensorrtInferenceEngine] enqueueV3 failed");
        return false;
    }

    // 拷贝输出到主机
    if (output_size > output_size_) {
        output_size = output_size_;
    }
    cudaMemcpyAsync(output_data, d_output_, output_size, cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(cuda_stream_));
    cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_));

    return true;
}

std::pair<int, int> TensorrtInferenceEngine::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int TensorrtInferenceEngine::getBatchSize() const {
    return config_.batch_size;
}

bool TensorrtInferenceEngine::isAvailable() const {
#ifdef WITH_TENSORRT
    return true;
#else
    return false;
#endif
}

void TensorrtInferenceEngine::setCudaStream(void* stream) {
    cuda_stream_ = stream;
}

void* TensorrtInferenceEngine::getRawContext() const {
    return context_.get();
}

bool TensorrtInferenceEngine::initEngine(const std::string& engine_path) {
    // 加载 .engine 文件
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        LOG_ERROR_FMT("[TensorrtInferenceEngine] Cannot open engine: {}", engine_path);
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        LOG_ERROR_FMT("[TensorrtInferenceEngine] Cannot read engine: {}", engine_path);
        return false;
    }

    // 创建 TensorRT runtime 和 engine
    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    if (!runtime_) {
        LOG_ERROR("[TensorrtInferenceEngine] Failed to create runtime");
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    if (!engine_) {
        LOG_ERROR("[TensorrtInferenceEngine] Failed to deserialize engine");
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        LOG_ERROR("[TensorrtInferenceEngine] Failed to create context");
        return false;
    }

    // 获取 tensor 名称
    int nb_io = engine_->getNbIOTensors();
    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name_ = name;
        } else {
            output_name_ = name;
        }
    }

    loaded_ = true;
    LOG_INFO_FMT("[TensorrtInferenceEngine] Engine loaded: {}", engine_path);
    return true;
}

bool TensorrtInferenceEngine::allocateBuffers() {
    // 计算输入输出大小
    nvinfer1::Dims input_dims = engine_->getTensorShape(input_name_.c_str());
    input_size_ = 1;
    for (int i = 0; i < input_dims.nbDims; ++i) {
        input_size_ *= (input_dims.d[i] > 0) ? input_dims.d[i] : 1;
    }
    input_size_ *= sizeof(float);

    nvinfer1::Dims output_dims = engine_->getTensorShape(output_name_.c_str());
    output_size_ = 1;
    for (int i = 0; i < output_dims.nbDims; ++i) {
        output_size_ *= (output_dims.d[i] > 0) ? output_dims.d[i] : 1;
    }
    output_size_ *= sizeof(float);

    // 分配 GPU 内存
    cudaMalloc(&d_input_, input_size_);
    cudaMalloc(&d_output_, output_size_);

    // 设置 tensor 地址
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);

    LOG_INFO_FMT("[TensorrtInferenceEngine] Buffers allocated: input={} bytes, output={} bytes",
                 input_size_, output_size_);
    return true;
}

void TensorrtInferenceEngine::freeBuffers() {
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
}

// 注册 TensorRT 后端到工厂
#ifdef WITH_TENSORRT
REGISTER_INFERENCE_BACKEND(InferenceBackend::TENSORRT, TensorrtInferenceEngine)
#endif

} // namespace hal
} // namespace ai_stream
