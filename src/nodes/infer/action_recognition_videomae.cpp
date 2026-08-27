// src/nodes/infer/action_recognition_videomae.cpp
#include "action_recognition_videomae.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "utils/time_util.h"
#include "3rd_party/log_mgr/log_mgr.h"

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
    // HAL 引擎自行管理 GPU 资源生命周期
    engine_.reset();
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

    hal::ActionResult hal_result;
    if (!engine_->inferGpuFrames(gpu_frames, hal_result)) {
        LOG_ERROR("[ActionRecognition] GPU clip inference failed");
        return;
    }
    result = toCoreResult(hal_result);
    result.timestamp_ms = utils::TimeUtil::currentTimeMs();
}

void ActionRecognitionVideoMAENode::inferFromCpu(const std::vector<cv::Mat>& clip,
                                                  core::InferenceResultPacket::ActionResult& result) {
    if (static_cast<int>(clip.size()) < cfg_.num_frames) return;

    // 预处理：布局转换 (HWC -> NCHW)，归一化已由上游预处理节点完成
    const size_t input_size = static_cast<size_t>(cfg_.batch_size) * clip.size() * 3 *
                              cfg_.input_height * cfg_.input_width * sizeof(float);
    std::vector<float> host_input(input_size / sizeof(float));
    preprocessClip(clip, host_input.data());

    hal::ActionResult hal_result;
    if (!engine_->inferPreprocessed(host_input.data(), input_size, hal_result)) {
        LOG_ERROR("[ActionRecognition] CPU clip inference failed");
        return;
    }
    result = toCoreResult(hal_result);
    result.timestamp_ms = utils::TimeUtil::currentTimeMs();
}

core::InferenceResultPacket::ActionResult ActionRecognitionVideoMAENode::toCoreResult(
    const hal::ActionResult& hal_result) const {
    core::InferenceResultPacket::ActionResult result;
    result.action_label = hal_result.action_label;
    result.confidence = hal_result.confidence;
    result.action_scores = hal_result.scores;
    return result;
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
    // 尝试从 JSON 配置文件加载参数（与 .engine 同名的 .json）
    std::string config_path = cfg_.model_path;
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
                    (void)num_classes;
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

    // 通过 HAL 工厂创建推理后端（TensorRT/RKNN/Ascend 按编译配置自动选择）
    engine_ = hal::ActionRecognitionFactory::instance().create(
        hal::ActionRecognitionBackend::AUTO);
    if (!engine_) {
        LOG_ERROR("[ActionRecognitionVideoMAE] No action recognition backend available "
                  "(check WITH_TENSORRT/WITH_RKNN/WITH_ASCEND build options)");
        return false;
    }

    hal::ActionRecognitionConfig hal_cfg;
    hal_cfg.model_path = cfg_.model_path;
    hal_cfg.input_height = cfg_.input_height;
    hal_cfg.input_width = cfg_.input_width;
    hal_cfg.num_frames = cfg_.num_frames;
    hal_cfg.frame_interval = cfg_.frame_interval;
    hal_cfg.batch_size = cfg_.batch_size;
    hal_cfg.action_labels = cfg_.action_labels;

    if (!engine_->loadModel(hal_cfg)) {
        LOG_ERROR_FMT("Failed to load model via HAL backend: {}", cfg_.model_path);
        engine_.reset();
        return false;
    }

    LOG_INFO_FMT("Action recognition model loaded successfully: {} (backend: {})",
                 cfg_.model_path, engine_->getBackendName());
    return true;
}

REGISTER_NODE("action_recognition_videomae", ActionRecognitionVideoMAENode)
} // namespace nodes
} // namespace ai_stream
