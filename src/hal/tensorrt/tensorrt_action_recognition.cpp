// src/hal/tensorrt/tensorrt_action_recognition.cpp
// TensorRT 动作识别引擎——封装现有 VideoMAE 推理逻辑到 HAL 接口
#include "tensorrt_action_recognition.h"
#include "ai_stream/hal/action_recognition_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"

#include <iostream>
#include <fstream>
#include <numeric>
#include <NvInfer.h>
#include <cuda_runtime_api.h>

namespace ai_stream {
namespace hal {

TensorrtActionRecognition::TensorrtActionRecognition() {
    LOG_DEBUG("[TensorrtActionRecognition] Constructor");
}

TensorrtActionRecognition::~TensorrtActionRecognition() {
    freeBuffers();
    LOG_DEBUG("[TensorrtActionRecognition] Destructor");
}

bool TensorrtActionRecognition::loadModel(const ActionRecognitionConfig& config) {
    config_ = config;
    return initEngine(config.model_path) && allocateBuffers();
}

bool TensorrtActionRecognition::infer(const uint8_t* clip_data, size_t clip_size,
                                       ActionResult& result) {
    if (!loaded_) {
        LOG_ERROR("[TensorrtActionRecognition] Model not loaded");
        return false;
    }

    // 预处理：将 clip 数据拷贝到 GPU 并归一化
    preprocessClip(clip_data, static_cast<float*>(d_input_));

    // 执行推理
    if (!context_->enqueueV3(static_cast<cudaStream_t>(cuda_stream_))) {
        LOG_ERROR("[TensorrtActionRecognition] enqueueV3 failed");
        return false;
    }

    // 拷贝输出到主机
    std::vector<float> output(output_size_ / sizeof(float));
    cudaMemcpyAsync(output.data(), d_output_, output_size_, cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(cuda_stream_));
    cudaStreamSynchronize(static_cast<cudaStream_t>(cuda_stream_));

    // 后处理
    result = postprocess(output.data(), output.size());
    return true;
}

std::pair<int, int> TensorrtActionRecognition::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int TensorrtActionRecognition::getNumFrames() const {
    return config_.num_frames;
}

bool TensorrtActionRecognition::isAvailable() const {
#ifdef WITH_TENSORRT
    return true;
#else
    return false;
#endif
}

void TensorrtActionRecognition::setCudaStream(void* stream) {
    cuda_stream_ = stream;
}

bool TensorrtActionRecognition::initEngine(const std::string& engine_path) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.good()) {
        LOG_ERROR_FMT("[TensorrtActionRecognition] Cannot open engine: {}", engine_path);
        return false;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        LOG_ERROR_FMT("[TensorrtActionRecognition] Cannot read engine: {}", engine_path);
        return false;
    }

    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    if (!runtime_) {
        LOG_ERROR("[TensorrtActionRecognition] Failed to create runtime");
        return false;
    }

    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), size));
    if (!engine_) {
        LOG_ERROR("[TensorrtActionRecognition] Failed to deserialize engine");
        return false;
    }

    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        LOG_ERROR("[TensorrtActionRecognition] Failed to create context");
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
    LOG_INFO_FMT("[TensorrtActionRecognition] Engine loaded: {}", engine_path);
    return true;
}

bool TensorrtActionRecognition::allocateBuffers() {
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

    cudaMalloc(&d_input_, input_size_);
    cudaMalloc(&d_output_, output_size_);

    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);

    LOG_INFO_FMT("[TensorrtActionRecognition] Buffers allocated: input={} bytes, output={} bytes",
                 input_size_, output_size_);
    return true;
}

void TensorrtActionRecognition::freeBuffers() {
    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
}

void TensorrtActionRecognition::preprocessClip(const uint8_t* clip_data, float* gpu_input) {
    // 实际实现：
    // 1. 从 clip_data 解析多帧图像
    // 2. 对每帧进行 resize、归一化
    // 3. 按 NCHW 布局排列（num_frames * 3 * H * W）
    // 4. 拷贝到 GPU
}

ActionResult TensorrtActionRecognition::postprocess(const float* output, int num_classes) {
    ActionResult result;

    // Softmax
    float max_val = *std::max_element(output, output + num_classes);
    std::vector<float> probs(num_classes);
    float sum = 0.0f;
    for (int i = 0; i < num_classes; ++i) {
        probs[i] = std::exp(output[i] - max_val);
        sum += probs[i];
    }
    for (int i = 0; i < num_classes; ++i) {
        probs[i] /= sum;
    }

    // Top-k
    std::vector<int> indices(num_classes);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&probs](int a, int b) {
        return probs[a] > probs[b];
    });

    result.action_id = indices[0];
    result.confidence = probs[indices[0]];
    if (result.action_id < static_cast<int>(config_.action_labels.size())) {
        result.action_label = config_.action_labels[result.action_id];
    }

    // Top-5
    for (int i = 0; i < std::min(5, num_classes); ++i) {
        int idx = indices[i];
        std::string label = (idx < static_cast<int>(config_.action_labels.size()))
                           ? config_.action_labels[idx] : "class_" + std::to_string(idx);
        result.top_k.emplace_back(label, probs[idx]);
    }

    return result;
}

// 注册 TensorRT 动作识别后端到工厂
#ifdef WITH_TENSORRT
REGISTER_ACTION_RECOGNITION_BACKEND(ActionRecognitionBackend::TENSORRT, TensorrtActionRecognition)
#endif

} // namespace hal
} // namespace ai_stream
