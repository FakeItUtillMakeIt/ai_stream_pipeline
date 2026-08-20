// src/nodes/infer/action_recognition_videomae.cpp
#include "action_recognition_videomae.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "utils/time_util.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/tensor_rt_logger.h"

#include <algorithm>
#include <numeric>
#include <fstream>
#include <nlohmann/json.hpp>


namespace ai_stream {
namespace nodes {

ActionRecognitionVideoMAENode::ActionRecognitionVideoMAENode()
    : core::QueuedNode<IActionRecognitionNode>("ActionRecognitionVideoMAE"), cfg_{} {
    // 默认配置：3分类（climbing, fighting, other）
    cfg_.action_labels = {"climbing", "fighting", "other"};
    cfg_.confidence_threshold = 0.7f;
    cfg_.num_frames = 16;
    cfg_.frame_interval = 2;
    cfg_.window_size = 16;
    cfg_.stride = 8;
    cfg_.input_height = 224;
    cfg_.input_width = 224;
    LOG_INFO_FMT("[ActionRecognitionVideoMAE] Default Constructor");
}

ActionRecognitionVideoMAENode::ActionRecognitionVideoMAENode(const Config& cfg)
    : core::QueuedNode<IActionRecognitionNode>("ActionRecognitionVideoMAE"), cfg_(cfg) {
    LOG_INFO_FMT("[ActionRecognitionVideoMAE] Constructor, model: {}, input_size: {}x{}, num_frames: {}, frame_interval: {}, window_size: {}, stride: {}, confidence_threshold: {}, batch_size: {}",
        cfg_.model_path, cfg_.input_width, cfg_.input_height, cfg_.num_frames, cfg_.frame_interval,
        cfg_.window_size, cfg_.stride, cfg_.confidence_threshold, cfg_.batch_size);
}

ActionRecognitionVideoMAENode::~ActionRecognitionVideoMAENode() {
    stop();
}

bool ActionRecognitionVideoMAENode::onStartup() {
    if (!loadModel()) {
        LOG_ERROR_FMT("Failed to load action recognition model");
        return false;
    }
    is_initialized_ = true;
    LOG_INFO_FMT("[ActionRecognitionVideoMAE] Started");
    return true;
}

void ActionRecognitionVideoMAENode::onShutdown() {
    // 释放GPU资源
    if (device_input_) {
        cudaFree(device_input_);
        device_input_ = nullptr;
    }
    if (device_output_) {
        cudaFree(device_output_);
        device_output_ = nullptr;
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    
    is_initialized_ = false;
    LOG_INFO_FMT("[ActionRecognitionVideoMAE] Stopped");
}

void ActionRecognitionVideoMAENode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO_FMT("[ActionRecognitionVideoMAE] Received stream end");
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 workerLoop 中检查，worker 线程会自然退出
        broadcast(packet);
        return;
    }
    if (!is_initialized_) return;
    
    in_time_ms_ = utils::TimeUtil::currentTimeMs();
    
    auto video_frame = std::dynamic_pointer_cast<core::VideoFramePacket>(packet);
    if (!video_frame) {
        broadcast(packet);
        return;
    }
    
    uint32_t stream_id = packet->stream_id;
    int64_t frame_id = packet->frame_id;
    
    // 根据数据类型选择处理路径
    if (video_frame->is_gpu && video_frame->d_ptr) {
        // GPU预处理路径：直接缓冲GPU数据
        updateGpuFrameBuffer(stream_id, video_frame->d_ptr, frame_id);
        
        if (shouldRunInference(stream_id)) {
            auto gpu_clip = getGpuClipFromBuffer(stream_id);
            
            if (gpu_clip.size() >= static_cast<size_t>(cfg_.num_frames)) {
                auto infer_result = std::make_shared<core::InferenceResultPacket>();
                infer_result->stream_id = stream_id;
                infer_result->frame_id = frame_id;
                infer_result->timestamp_ms = packet->timestamp_ms;
                infer_result->source_frame = video_frame;
                
                core::InferenceResultPacket::ActionResult action_result;
                inferFromGpu(gpu_clip, action_result);
                
                if (action_result.confidence >= cfg_.confidence_threshold) {
                    infer_result->action_results.push_back(action_result);
                    LOG_INFO_FMT("[ActionRecognition] Detected action: {} (confidence: {:.4f})", 
                                 action_result.action_label, action_result.confidence);
                }
                
                uint64_t cost = utils::TimeUtil::currentTimeMs() - in_time_ms_;
                infer_result->cost_ms = cost;
                infer_result->cost_time_map[name_] = cost;
                
                broadcast(infer_result);
            }
        }
    } else if (video_frame->mat && !video_frame->mat->empty()) {
        // CPU预处理路径：预处理节点已完成resize+normalize，只需布局转换
        updateFrameBuffer(stream_id, *video_frame->mat, frame_id);
        
        if (shouldRunInference(stream_id)) {
            auto clip = getClipFromBuffer(stream_id);
            
            if (clip.size() >= static_cast<size_t>(cfg_.num_frames)) {
                auto infer_result = std::make_shared<core::InferenceResultPacket>();
                infer_result->stream_id = stream_id;
                infer_result->frame_id = frame_id;
                infer_result->timestamp_ms = packet->timestamp_ms;
                infer_result->source_frame = video_frame;
                
                core::InferenceResultPacket::ActionResult action_result;
                inferFromCpu(clip, action_result);
                
                if (action_result.confidence >= cfg_.confidence_threshold) {
                    infer_result->action_results.push_back(action_result);
                    LOG_INFO_FMT("[ActionRecognition] Detected action: {} (confidence: {:.4f})", 
                                 action_result.action_label, action_result.confidence);
                }
                
                uint64_t cost = utils::TimeUtil::currentTimeMs() - in_time_ms_;
                infer_result->cost_ms = cost;
                infer_result->cost_time_map[name_] = cost;
                
                broadcast(infer_result);
            }
        }
    }
    
    // 广播原始packet（用于下游节点）
    broadcast(packet);
}

bool ActionRecognitionVideoMAENode::shouldRunInference(uint32_t stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& state = window_states_[stream_id];
    state.frame_counter++;
    
    // 检查是否达到滑动步长
    if (state.frame_counter >= cfg_.stride) {
        state.frame_counter = 0;
        // 根据使用的缓冲区获取最后帧ID
        if (!gpu_frame_buffers_[stream_id].frames.empty()) {
            state.last_inference_frame = gpu_frame_buffers_[stream_id].frames.back().first;
        } else if (!frame_buffers_[stream_id].frames.empty()) {
            state.last_inference_frame = frame_buffers_[stream_id].frames.back().first;
        }
        return true;
    }
    
    return false;
}

void ActionRecognitionVideoMAENode::updateGpuFrameBuffer(uint32_t stream_id, 
                                                          void* d_ptr, 
                                                          int64_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& buffer = gpu_frame_buffers_[stream_id];
    buffer.frames.push_back({frame_id, d_ptr});
    
    // 限制缓冲区大小
    int max_buffer_size = cfg_.window_size * cfg_.frame_interval + 10;
    while (static_cast<int>(buffer.frames.size()) > max_buffer_size) {
        buffer.frames.pop_front();
    }
}

void ActionRecognitionVideoMAENode::updateFrameBuffer(uint32_t stream_id, 
                                                       const cv::Mat& frame, 
                                                       int64_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& buffer = frame_buffers_[stream_id];
    buffer.frames.push_back({frame_id, frame.clone()});
    
    // 限制缓冲区大小
    int max_buffer_size = cfg_.window_size * cfg_.frame_interval + 10;
    while (static_cast<int>(buffer.frames.size()) > max_buffer_size) {
        buffer.frames.pop_front();
    }
}

std::vector<void*> ActionRecognitionVideoMAENode::getGpuClipFromBuffer(uint32_t stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<void*> clip;
    auto& buffer = gpu_frame_buffers_[stream_id];
    
    if (buffer.frames.empty()) return clip;
    
    // 按间隔采样帧
    int step = cfg_.frame_interval;
    
    // 从最新的帧开始，向前采样
    for (int i = static_cast<int>(buffer.frames.size()) - 1; 
         i >= 0 && static_cast<int>(clip.size()) < cfg_.num_frames; 
         i -= step) {
        clip.push_back(buffer.frames[i].second);
    }
    
    // 反转clip，使其按时间顺序排列
    std::reverse(clip.begin(), clip.end());
    
    return clip;
}

std::vector<cv::Mat> ActionRecognitionVideoMAENode::getClipFromBuffer(uint32_t stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::vector<cv::Mat> clip;
    auto& buffer = frame_buffers_[stream_id];
    
    if (buffer.frames.empty()) return clip;
    
    // 按间隔采样帧
    int step = cfg_.frame_interval;
    
    // 从最新的帧开始，向前采样
    for (int i = static_cast<int>(buffer.frames.size()) - 1; 
         i >= 0 && static_cast<int>(clip.size()) < cfg_.num_frames; 
         i -= step) {
        clip.push_back(buffer.frames[i].second);
    }
    
    // 反转clip，使其按时间顺序排列
    std::reverse(clip.begin(), clip.end());
    
    return clip;
}

void ActionRecognitionVideoMAENode::inferFromGpu(const std::vector<void*>& gpu_frames, 
                                                  core::InferenceResultPacket::ActionResult& result) {
    if (static_cast<int>(gpu_frames.size()) < cfg_.num_frames) return;
    
    // 每帧数据大小: 3 * height * width * sizeof(float) = 3 * 224 * 224 * 4 = 602112 bytes
    int single_frame_size = 3 * cfg_.input_height * cfg_.input_width * sizeof(float);
    
    // 将16帧GPU数据拷贝到连续的输入缓冲区
    // 布局: [1, 16, 3, 224, 224] = [batch, frames, channels, height, width]
    for (int i = 0; i < cfg_.num_frames; i++) {
        void* dst = static_cast<char*>(device_input_) + i * single_frame_size;
        // 每帧已经由cuda_resize_normalize预处理为NCHW格式
        // 直接拷贝到输入缓冲区的对应位置
        cudaMemcpyAsync(dst, gpu_frames[i], single_frame_size, 
                        cudaMemcpyDeviceToDevice, stream_);
    }
    cudaStreamSynchronize(stream_);
    
    // 设置输入形状
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 5;
    input_dims.d[0] = cfg_.batch_size;
    input_dims.d[1] = cfg_.num_frames;
    input_dims.d[2] = 3;
    input_dims.d[3] = cfg_.input_height;
    input_dims.d[4] = cfg_.input_width;
    context_->setInputShape("input", input_dims);
    
    // 推理
    if (!context_->enqueueV3(stream_)) {
        spdlog::error("enqueueV3 failed for action recognition");
        return;
    }
    cudaStreamSynchronize(stream_);
    
    // 拷贝输出
    std::vector<float> host_output(output_size_ / sizeof(float));
    cudaMemcpyAsync(host_output.data(), device_output_, output_size_, 
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    
    // 后处理
    result = postprocess(host_output.data(), static_cast<int>(cfg_.action_labels.size()));
    result.timestamp_ms = utils::TimeUtil::currentTimeMs();
}

void ActionRecognitionVideoMAENode::inferFromCpu(const std::vector<cv::Mat>& clip, 
                                                  core::InferenceResultPacket::ActionResult& result) {
    if (static_cast<int>(clip.size()) < cfg_.num_frames) return;
    
    // 预处理：布局转换 (HWC -> NCHW) + 归一化
    std::vector<float> host_input(input_size_ / sizeof(float));
    preprocessClip(clip, host_input.data());
    
    // 拷贝到GPU
    cudaError_t err = cudaMemcpyAsync(device_input_, host_input.data(), input_size_, 
                    cudaMemcpyHostToDevice, stream_);
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("cudaMemcpy H2D failed: {}", cudaGetErrorString(err));
        return;
    }
    cudaStreamSynchronize(stream_);
    
    // 设置输入形状
    nvinfer1::Dims input_dims;
    input_dims.nbDims = 5;
    input_dims.d[0] = cfg_.batch_size;
    input_dims.d[1] = cfg_.num_frames;
    input_dims.d[2] = 3;
    input_dims.d[3] = cfg_.input_height;
    input_dims.d[4] = cfg_.input_width;
    context_->setInputShape("input", input_dims);
    
    // 推理
    if (!context_->enqueueV3(stream_)) {
        spdlog::error("enqueueV3 failed for action recognition");
        return;
    }
    cudaStreamSynchronize(stream_);
    
    // 拷贝输出
    std::vector<float> host_output(output_size_ / sizeof(float));
    err = cudaMemcpyAsync(host_output.data(), device_output_, output_size_, 
                    cudaMemcpyDeviceToHost, stream_);
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("cudaMemcpy D2H failed: {}", cudaGetErrorString(err));
        return;
    }
    cudaStreamSynchronize(stream_);
    
    // 后处理
    result = postprocess(host_output.data(), static_cast<int>(cfg_.action_labels.size()));
    result.timestamp_ms = utils::TimeUtil::currentTimeMs();
}

void ActionRecognitionVideoMAENode::preprocessClip(const std::vector<cv::Mat>& clip, 
                                                    float* input_buffer) {
    // 假设输入已经由预处理节点完成resize+normalize
    // 这里只做布局转换：HWC -> NCHW
    
    int h = cfg_.input_height;
    int w = cfg_.input_width;
    int num_frames = static_cast<int>(clip.size());
    
    for (int t = 0; t < num_frames; t++) {
        // 输入是HWC格式的float数据（已归一化）
        const float* src = reinterpret_cast<const float*>(clip[t].data);
        
        // 转换为NCHW格式
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    input_buffer[t * 3 * h * w + c * h * w + i * w + j] = 
                        src[i * w * 3 + j * 3 + c];
                }
            }
        }
    }
}

core::InferenceResultPacket::ActionResult ActionRecognitionVideoMAENode::postprocess(
    const float* output, int num_classes) {
    core::InferenceResultPacket::ActionResult result;
    
    // Softmax
    std::vector<float> scores(num_classes);
    float max_score = *std::max_element(output, output + num_classes);
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        scores[i] = std::exp(output[i] - max_score);
        sum += scores[i];
    }
    for (int i = 0; i < num_classes; i++) {
        scores[i] /= sum;
    }
    
    // 获取最大分数
    auto max_it = std::max_element(scores.begin(), scores.end());
    int max_idx = static_cast<int>(std::distance(scores.begin(), max_it));
    
    result.confidence = *max_it;
    result.action_scores = scores;
    
    if (max_idx < static_cast<int>(cfg_.action_labels.size())) {
        result.action_label = cfg_.action_labels[max_idx];
    }
    
    return result;
}

// 接口实现
void ActionRecognitionVideoMAENode::setModelPath(const std::string& model_path) {
    cfg_.model_path = model_path;
}

void ActionRecognitionVideoMAENode::setInputSize(int height, int width) {
    cfg_.input_height = height;
    cfg_.input_width = width;
}

void ActionRecognitionVideoMAENode::setClipParams(int num_frames, int frame_interval) {
    cfg_.num_frames = num_frames;
    cfg_.frame_interval = frame_interval;
}

void ActionRecognitionVideoMAENode::setSlidingWindow(int window_size, int stride) {
    cfg_.window_size = window_size;
    cfg_.stride = stride;
}

void ActionRecognitionVideoMAENode::setActionLabels(const std::vector<std::string>& labels) {
    cfg_.action_labels = labels;
}

void ActionRecognitionVideoMAENode::setConfidenceThreshold(float threshold) {
    cfg_.confidence_threshold = threshold;
}

void ActionRecognitionVideoMAENode::setBatchSize(int batch_size) {
    cfg_.batch_size = batch_size;
}

bool ActionRecognitionVideoMAENode::loadModel() {
    // 尝试从JSON配置文件加载参数
    std::string config_path = cfg_.model_path;
    // 将.engine替换为.json查找配置文件
    auto pos = config_path.find(".engine");
    if (pos != std::string::npos) {
        config_path.replace(pos, 6, ".json");
        std::ifstream config_file(config_path);
        if (config_file.is_open()) {
            try {
                nlohmann::json config;
                config_file >> config;
                
                if (config.contains("confidence_threshold")) {
                    cfg_.confidence_threshold = config["confidence_threshold"].get<float>();
                }
                if (config.contains("num_classes")) {
                    int num_classes = config["num_classes"].get<int>();
                    if (config.contains("class_names")) {
                        cfg_.action_labels = config["class_names"].get<std::vector<std::string>>();
                    }
                }
                if (config.contains("num_frames")) {
                    cfg_.num_frames = config["num_frames"].get<int>();
                }
                if (config.contains("frame_stride")) {
                    cfg_.stride = config["frame_stride"].get<int>();
                }
                LOG_INFO_FMT("Loaded config from: {}", config_path);
            } catch (const std::exception& e) {
                LOG_WARN_FMT("Failed to parse config file: {}, using defaults", config_path);
            }
        }
    }
    
    // 加载TensorRT引擎
    std::ifstream file(cfg_.model_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        LOG_ERROR_FMT("Failed to open model file: {}", cfg_.model_path);
        return false;
    }
    
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        LOG_ERROR_FMT("Failed to read model file: {}", cfg_.model_path);
        return false;
    }
    
    runtime_.reset(nvinfer1::createInferRuntime(g_logger));
    if (!runtime_) {
        LOG_ERROR_FMT("Failed to create TensorRT runtime");
        return false;
    }
    
    engine_.reset(runtime_->deserializeCudaEngine(buffer.data(), static_cast<size_t>(size)));
    if (!engine_) {
        LOG_ERROR_FMT("Failed to deserialize TensorRT engine");
        return false;
    }
    
    context_.reset(engine_->createExecutionContext());
    if (!context_) {
        LOG_ERROR_FMT("Failed to create TensorRT execution context");
        return false;
    }
    
    // 创建CUDA流
    cudaStreamCreate(&stream_);
    
    // 分配缓冲区
    allocateBuffers();
    
    LOG_INFO_FMT("Action recognition model loaded successfully: {}", cfg_.model_path);
    return true;
}

void ActionRecognitionVideoMAENode::allocateBuffers() {
    // 获取输入输出tensor名称
    int nb_io = engine_->getNbIOTensors();
    LOG_INFO_FMT("[ActionRecognition] Engine has {} I/O tensors", nb_io);
    
    const char* input_name = nullptr;
    const char* output_name = nullptr;
    
    for (int i = 0; i < nb_io; ++i) {
        const char* name = engine_->getIOTensorName(i);
        nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
        nvinfer1::Dims dims = engine_->getTensorShape(name);
        std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
        
        // 计算缓冲区大小 (处理动态维度-1)
        size_t buf_size = 1;
        for (int d = 0; d < dims.nbDims; ++d) {
            buf_size *= (dims.d[d] > 0) ? dims.d[d] : 1;  // -1 替换为 1
        }
        buf_size *= sizeof(float);
        
        LOG_INFO_FMT("[ActionRecognition]  Tensor[{}]: {} ({}), nbDims={}, dims=[{},{},{},{},{}], size={}bytes",
                     i, name, mode_str, dims.nbDims, 
                     dims.d[0], dims.d[1], dims.d[2], dims.d[3], dims.d[4], buf_size);
        
        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_name = name;
            input_size_ = buf_size;
        } else {
            output_name = name;
            output_size_ = buf_size;
        }
    }
    
    // 对于动态batch，使用实际batch大小重新计算
    // 输入: [batch, 16, 3, 224, 224]
    input_size_ = cfg_.batch_size * 16 * 3 * cfg_.input_height * cfg_.input_width * sizeof(float);
    // 输出: [batch, 3]
    output_size_ = cfg_.batch_size * cfg_.action_labels.size() * sizeof(float);
    
    LOG_INFO_FMT("[ActionRecognition] Input name: {}, size: {} bytes", input_name, input_size_);
    LOG_INFO_FMT("[ActionRecognition] Output name: {}, size: {} bytes", output_name, output_size_);
    
    // 分配GPU内存
    cudaMalloc(&device_input_, input_size_);
    cudaMalloc(&device_output_, output_size_);
    
    // 绑定缓冲区
    context_->setTensorAddress(input_name, device_input_);
    context_->setTensorAddress(output_name, device_output_);
    
    LOG_INFO_FMT("Input buffer size: {} bytes", input_size_);
    LOG_INFO_FMT("Output buffer size: {} bytes", output_size_);
}
REGISTER_NODE("action_recognition_videomae", ActionRecognitionVideoMAENode)
} // namespace nodes
} // namespace ai_stream
