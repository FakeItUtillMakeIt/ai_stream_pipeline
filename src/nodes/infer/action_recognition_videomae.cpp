// src/nodes/infer/action_recognition_videomae.cpp
#include "action_recognition_videomae.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "utils/time_util.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "tensor_rt_logger.h"

#include <algorithm>
#include <numeric>
#include <fstream>


namespace ai_stream {
namespace nodes {

ActionRecognitionVideoMAENode::ActionRecognitionVideoMAENode()
    : cfg_{} {
    name_ = "ActionRecognitionVideoMAE";
    LOG_INFO_FMT("[ActionRecognitionVideoMAE] Default Constructor");
}

ActionRecognitionVideoMAENode::ActionRecognitionVideoMAENode(const Config& cfg)
    : cfg_(cfg) {
    name_ = "ActionRecognitionVideoMAE";
    LOG_INFO_FMT("[ActionRecognitionVideoMAE] Constructor, model: {}, input_size: {}x{}, num_frames: {}, frame_interval: {}, window_size: {}, stride: {}, confidence_threshold: {}, batch_size: {}",
        cfg_.model_path, cfg_.input_width, cfg_.input_height, cfg_.num_frames, cfg_.frame_interval,
        cfg_.window_size, cfg_.stride, cfg_.confidence_threshold, cfg_.batch_size);
}

ActionRecognitionVideoMAENode::~ActionRecognitionVideoMAENode() {
    stop();
}

bool ActionRecognitionVideoMAENode::start() {
    if (!loadModel()) {
        LOG_ERROR_FMT("Failed to load action recognition model");
        return false;
    }
    is_initialized_ = true;
    running_ = true;
    LOG_INFO_FMT("[ActionRecognitionVideoMAE] Started");
    return true;
}

void ActionRecognitionVideoMAENode::stop() {
    running_ = false;
    
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

void ActionRecognitionVideoMAENode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (!is_initialized_) return;
    
    in_time_ms_ = utils::TimeUtil::currentTimeMs();
    
    auto video_frame = std::dynamic_pointer_cast<core::VideoFramePacket>(packet);
    if (!video_frame || !video_frame->mat) {
        broadcast(packet);
        return;
    }
    
    uint32_t stream_id = packet->stream_id;
    int64_t frame_id = packet->frame_id;
    
    // 更新帧缓冲区
    updateFrameBuffer(stream_id, *video_frame->mat, frame_id);
    
    // 检查是否应该执行推理
    if (shouldRunInference(stream_id)) {
        // 获取clip
        auto clip = getClipFromBuffer(stream_id);
        
        if (clip.size() >= static_cast<size_t>(cfg_.num_frames)) {
            // 执行推理
            auto infer_result = std::make_shared<core::InferenceResultPacket>();
            infer_result->stream_id = stream_id;
            infer_result->frame_id = frame_id;
            infer_result->timestamp_ms = packet->timestamp_ms;
            infer_result->source_frame = video_frame;
            
            core::InferenceResultPacket::ActionResult action_result;
            infer(clip, action_result);
            
            if (action_result.confidence >= cfg_.confidence_threshold) {
                infer_result->action_results.push_back(action_result);
            }
            
            // 记录耗时
            uint64_t cost = utils::TimeUtil::currentTimeMs() - in_time_ms_;
            infer_result->cost_ms = cost;
            infer_result->cost_time_map[name_] = cost;
            
            broadcast(infer_result);
        }
    }
    
    // 广播原始packet（用于下游节点）
    broadcast(packet);
}

void ActionRecognitionVideoMAENode::updateFrameBuffer(uint32_t stream_id, 
                                                       const cv::Mat& frame, 
                                                       int64_t frame_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& buffer = frame_buffers_[stream_id];
    buffer.frames.push_back({frame_id, frame.clone()});
    
    // 限制缓冲区大小（保留最近的帧）
    int max_buffer_size = cfg_.window_size * cfg_.frame_interval + 10;
    while (static_cast<int>(buffer.frames.size()) > max_buffer_size) {
        buffer.frames.pop_front();
    }
}

bool ActionRecognitionVideoMAENode::shouldRunInference(uint32_t stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& state = window_states_[stream_id];
    state.frame_counter++;
    
    // 检查是否达到滑动步长
    if (state.frame_counter >= cfg_.stride) {
        state.frame_counter = 0;
        if (!frame_buffers_[stream_id].frames.empty()) {
            state.last_inference_frame = frame_buffers_[stream_id].frames.back().first;
        }
        return true;
    }
    
    return false;
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

void ActionRecognitionVideoMAENode::preprocessClip(const std::vector<cv::Mat>& clip, 
                                                    float* input_buffer) {
    // VideoMAE预处理：resize + normalize
    // 输入格式：[1, num_frames*3, height, width] (NCHW, 视频展开为通道)
    
    int h = cfg_.input_height;
    int w = cfg_.input_width;
    int num_frames = static_cast<int>(clip.size());
    
    // ImageNet标准化参数
    float mean[3] = {0.485f, 0.456f, 0.406f};
    float std[3] = {0.229f, 0.224f, 0.225f};
    
    for (int t = 0; t < num_frames; t++) {
        cv::Mat resized;
        cv::resize(clip[t], resized, cv::Size(w, h));
        
        for (int c = 0; c < 3; c++) {
            for (int i = 0; i < h; i++) {
                for (int j = 0; j < w; j++) {
                    float pixel = resized.at<cv::Vec3b>(i, j)[c] / 255.0f;
                    input_buffer[t * 3 * h * w + c * h * w + i * w + j] = 
                        (pixel - mean[c]) / std[c];
                }
            }
        }
    }
}

void ActionRecognitionVideoMAENode::infer(const std::vector<cv::Mat>& clip, 
                                           core::InferenceResultPacket::ActionResult& result) {
    if (static_cast<int>(clip.size()) < cfg_.num_frames) return;
    
    // 预处理
    std::vector<float> host_input(input_size_ / sizeof(float));
    preprocessClip(clip, host_input.data());
    
    // 拷贝到GPU
    cudaMemcpyAsync(device_input_, host_input.data(), input_size_, 
                    cudaMemcpyHostToDevice, stream_);
    
    // 推理
    if (!context_->enqueueV3(stream_)) {
        spdlog::error("enqueueV3 failed for action recognition");
        return;
    }
    
    // 拷贝输出
    std::vector<float> host_output(output_size_ / sizeof(float));
    cudaMemcpyAsync(host_output.data(), device_output_, output_size_, 
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);
    
    // 后处理
    result = postprocess(host_output.data(), static_cast<int>(cfg_.action_labels.size()));
    result.timestamp_ms = utils::TimeUtil::currentTimeMs();
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
    const char* input_name = engine_->getIOTensorName(0);
    const char* output_name = engine_->getIOTensorName(1);
    
    // 获取输入输出形状
    auto input_shape = context_->getTensorShape(input_name);
    auto output_shape = context_->getTensorShape(output_name);
    
    // 计算输入大小
    input_size_ = 1;
    for (int i = 0; i < input_shape.d[0]; i++) {
        input_size_ *= static_cast<size_t>(input_shape.d[i]);
    }
    input_size_ *= sizeof(float);
    
    // 计算输出大小
    output_size_ = 1;
    for (int i = 0; i < output_shape.d[0]; i++) {
        output_size_ *= static_cast<size_t>(output_shape.d[i]);
    }
    output_size_ *= sizeof(float);
    
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
