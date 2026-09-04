// src/hal/tensorrt/tensorrt_action_recognition.cpp
#include "tensorrt_action_recognition.h"
#include "ai_stream/hal/action_recognition_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"

#include <algorithm>
#include <numeric>
#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <NvInfer.h>

namespace ai_stream {
namespace hal {

TensorrtActionRecognition::TensorrtActionRecognition() {
    LOG_DEBUG("[TensorrtActionRecognition] Constructor");
}

TensorrtActionRecognition::~TensorrtActionRecognition() {
    LOG_DEBUG("[TensorrtActionRecognition] Destructor");
}

bool TensorrtActionRecognition::loadModel(const ActionRecognitionConfig& config) {
    config_ = config;

    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        config.device_id < 0 || config.device_id >= device_count ||
        cudaSetDevice(config.device_id) != cudaSuccess) {
        LOG_ERROR_FMT("[TensorrtActionRecognition] Cannot select CUDA device {}", config.device_id);
        return false;
    }

    if (!core_.loadEngine(config.model_path, "TensorrtActionRecognition")) {
        return false;
    }

    // 确定 IO tensor 名称（第一个输入/第一个输出）
    bool output_found = false;
    for (const auto& meta : core_.tensors()) {
        if (meta.is_input) {
            input_name_ = meta.name;
        } else if (!output_found) {
            output_name_ = meta.name;
            output_found = true;
        }
    }

    // 按实际配置计算缓冲区大小（与 enqueueAndPostprocess 设置的形状一致）；
    // 输出取引擎推导值与配置值中的较大者，避免动态维度被替换为 1 后偏小
    const size_t in_by_config =
        static_cast<size_t>(config_.batch_size) * config_.num_frames * 3 *
        config_.input_height * config_.input_width * sizeof(float);
    size_t out_by_engine = 0;
    for (const auto& meta : core_.tensors()) {
        if (!meta.is_input && meta.name == output_name_) {
            out_by_engine = meta.elementsIgnoringDynamic() * sizeof(float);
        }
    }
    const size_t out_by_config = static_cast<size_t>(config_.batch_size) *
                                 std::max<size_t>(config_.action_labels.size(), 1) *
                                 sizeof(float);

    if (!core_.allocBuffer(input_name_, std::max(in_by_config, static_cast<size_t>(1)))) {
        return false;
    }
    if (!core_.allocBuffer(output_name_, std::max(std::max(out_by_engine, out_by_config),
                                                  static_cast<size_t>(1)))) {
        return false;
    }

    loaded_ = true;
    LOG_INFO_FMT("[TensorrtActionRecognition] Model loaded: {} (input={}, output={})",
                 config.model_path,
                 core_.bufferSize(input_name_), core_.bufferSize(output_name_));
    return true;
}

bool TensorrtActionRecognition::infer(const uint8_t* clip_data, size_t clip_size,
                                       ActionResult& result) {
    if (!loaded_) {
        LOG_ERROR("[TensorrtActionRecognition] Model not loaded");
        return false;
    }

    // 预处理：将 clip 数据拷贝到 GPU 并归一化
    void* d_input = core_.allocBuffer(input_name_, core_.bufferSize(input_name_));
    preprocessClip(clip_data, static_cast<float*>(d_input));

    // 执行推理
    return enqueueAndPostprocess(result);
}

bool TensorrtActionRecognition::inferPreprocessed(const float* input_nchw,
                                                  size_t size_bytes,
                                                  ActionResult& result) {
    if (!loaded_) {
        LOG_ERROR("[TensorrtActionRecognition] Model not loaded");
        return false;
    }
    if (!input_nchw || size_bytes == 0) return false;

    const size_t copy_size = std::min(size_bytes, core_.bufferSize(input_name_));
    void* d_input = core_.buffer(input_name_);
    cudaError_t err = cudaMemcpyAsync(d_input, input_nchw, copy_size,
                                      cudaMemcpyHostToDevice,
                                      static_cast<cudaStream_t>(core_.stream()));
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("[TensorrtActionRecognition] cudaMemcpy H2D failed: {}",
                      cudaGetErrorString(err));
        return false;
    }
    core_.synchronize();

    return enqueueAndPostprocess(result);
}

bool TensorrtActionRecognition::inferGpuFrames(const std::vector<void*>& gpu_frames,
                                               ActionResult& result) {
    if (!loaded_) {
        LOG_ERROR("[TensorrtActionRecognition] Model not loaded");
        return false;
    }
    if (static_cast<int>(gpu_frames.size()) < config_.num_frames) return false;

    auto stream = static_cast<cudaStream_t>(core_.stream());
    void* d_input = core_.buffer(input_name_);

    // 每帧已由 GPU 预处理节点输出为 NCHW float 设备内存
    const size_t single_frame_size =
        3 * config_.input_height * config_.input_width * sizeof(float);

    // 将 num_frames 帧 D2D 拼接为连续输入 [batch, frames, 3, H, W]
    for (int i = 0; i < config_.num_frames; i++) {
        char* dst = static_cast<char*>(d_input) + i * single_frame_size;
        cudaError_t err = cudaMemcpyAsync(dst, gpu_frames[i], single_frame_size,
                                          cudaMemcpyDeviceToDevice, stream);
        if (err != cudaSuccess) {
            LOG_ERROR_FMT("[TensorrtActionRecognition] cudaMemcpy D2D failed: {}",
                          cudaGetErrorString(err));
            return false;
        }
    }
    core_.synchronize();

    return enqueueAndPostprocess(result);
}

std::pair<int, int> TensorrtActionRecognition::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int TensorrtActionRecognition::getNumFrames() const {
    return config_.num_frames;
}

bool TensorrtActionRecognition::isAvailable() const {
#ifdef WITH_TENSORRT
    int device_count = 0;
    return cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0;
#else
    return false;
#endif
}

void TensorrtActionRecognition::setCudaStream(void* stream) {
    core_.setStream(stream);
}

bool TensorrtActionRecognition::enqueueAndPostprocess(ActionResult& result) {
    // 动态输入形状 [batch, frames, 3, H, W]
    int64_t dims[5] = {config_.batch_size, config_.num_frames, 3,
                       config_.input_height, config_.input_width};
    if (!core_.setInputShape(input_name_, dims, 5)) {
        LOG_ERROR("[TensorrtActionRecognition] setInputShape failed");
        return false;
    }

    if (!core_.enqueue()) {
        LOG_ERROR("[TensorrtActionRecognition] enqueueV3 failed");
        return false;
    }

    std::vector<float> output(core_.bufferSize(output_name_) / sizeof(float));
    cudaMemcpyAsync(output.data(), core_.buffer(output_name_),
                    core_.bufferSize(output_name_), cudaMemcpyDeviceToHost,
                    static_cast<cudaStream_t>(core_.stream()));
    core_.synchronize();

    // 类别数以引擎输出为准，但不超过已知标签数（防止动态维度放大）
    int num_classes = static_cast<int>(output.size());
    if (!config_.action_labels.empty()) {
        num_classes = std::min<int>(num_classes,
                                    static_cast<int>(config_.action_labels.size()));
    }
    result = postprocess(output.data(), num_classes);
    return true;
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
    result.scores = probs;
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
