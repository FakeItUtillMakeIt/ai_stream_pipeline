// src/nodes/infer/tensorrt_infer.cpp
#include "tensorrt_infer.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>

// 包含完整的 TensorRT 头文件
#include <NvInfer.h>

namespace ai_stream {
namespace nodes {

// ============================================================
// TensorRT Logger 类
// ============================================================
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        // 过滤掉 INFO 级别的消息，减少日志量
        if (severity == Severity::kINFO) return;
        
        switch (severity) {
            case Severity::kINTERNAL_ERROR:
                LOG_ERROR_FMT("[TensorRT] INTERNAL_ERROR: {}", msg);
                break;
            case Severity::kERROR:
                LOG_ERROR_FMT("[TensorRT] ERROR: {}", msg);
                break;
            case Severity::kWARNING:
                LOG_WARN_FMT("[TensorRT] WARNING: {}", msg);
                break;
            case Severity::kINFO:
                LOG_INFO_FMT("[TensorRT] INFO: {}", msg);
                break;
            case Severity::kVERBOSE:
                LOG_DEBUG_FMT("[TensorRT] VERBOSE: {}", msg);
                break;
            default:
                LOG_DEBUG_FMT("[TensorRT] UNKNOWN: {}", msg);
                break;
        }
    }
};

// 全局 Logger 实例
static TensorRTLogger g_logger;

// 构造函数
TensorRTInferNode::TensorRTInferNode() : IInferNode("TensorRTInfer") {
    LOG_DEBUG_FMT("[TensorRTInfer] Constructor");
}

// 析构函数
TensorRTInferNode::~TensorRTInferNode() {
    stop();
    LOG_DEBUG_FMT("[TensorRTInfer] Destructor");
}

bool TensorRTInferNode::loadModel(const std::string& model_path) {
    LOG_INFO_FMT("[TensorRTInfer] Loading model from: {}", model_path);
    
    // 检查文件是否存在
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR_FMT("[TensorRTInfer] Model file not found: {}", model_path);
        return false;
    }
    
    return initEngine(model_path);
}

void TensorRTInferNode::setPrecision(const std::string& precision) {
    LOG_INFO_FMT("[TensorRTInfer] Set precision: {}", precision);
}

void TensorRTInferNode::setBatchSize(int batch_size) {
    batch_size_ = batch_size;
    LOG_INFO_FMT("[TensorRTInfer] Set batch size: {}", batch_size);
}

std::pair<int, int> TensorRTInferNode::getInputSize() const {
    return {input_width_, input_height_};
}

std::vector<std::string> TensorRTInferNode::getClassNames() const {
    return class_names_;
}

bool TensorRTInferNode::start() {
    // 即使没有模型也允许启动（用于测试）
    if (!engine_) {
        LOG_WARN_FMT("[TensorRTInfer] No model loaded, will use mock inference");
    }
    running_ = true;
    worker_ = std::thread(&TensorRTInferNode::inferLoop, this);
    LOG_INFO_FMT("[TensorRTInfer] Started");
    return true;
}

void TensorRTInferNode::stop() {
    running_ = false;
    queue_.stop();
    if (worker_.joinable()) {
        worker_.join();
    }
    LOG_INFO_FMT("[TensorRTInfer] Stopped");
}

void TensorRTInferNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (!running_) return;
    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::dynamic_pointer_cast<core::VideoFramePacket>(packet);
    if (frame) {
        queue_.push(frame, std::chrono::milliseconds(10));
    }
}

void TensorRTInferNode::inferLoop() {
    while (running_) {
        std::shared_ptr<core::VideoFramePacket> frame;
        if (!queue_.pop(frame, std::chrono::milliseconds(100))) {
            continue;
        }
        
        auto result = std::make_shared<core::InferenceResultPacket>();
        result->stream_id = frame->stream_id;
        result->timestamp_ms = frame->timestamp_ms;
        result->source_frame = frame;
        
        if (engine_ && context_) {
            // TODO: 真实推理
            LOG_DEBUG_FMT("[TensorRTInfer] Performing real inference");
        } else {
            // 模拟推理（用于测试）
            static int frame_count = 0;
            frame_count++;
            
            core::InferenceResultPacket::BBox box;
            box.x = 100 + (frame_count % 200);
            box.y = 100 + ((frame_count / 2) % 200);
            box.w = 200;
            box.h = 300;
            box.confidence = 0.85f;
            box.class_id = frame_count % (class_names_.empty() ? 5 : class_names_.size());
            if (!class_names_.empty()) {
                box.class_name = class_names_[box.class_id];
            } else {
                box.class_name = "unknown";
            }
            result->detections.push_back(box);
        }
        
        broadcast(result);
    }
}

bool TensorRTInferNode::initEngine(const std::string& engine_path) {
    try {
        // 读取引擎文件
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR_FMT("[TensorRTInfer] Failed to open engine file: {}", engine_path);
            return false;
        }
        
        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            LOG_ERROR_FMT("[TensorRTInfer] Failed to read engine file");
            return false;
        }
        
        // 创建运行时和引擎 - 使用全局 Logger
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
        if (!runtime) {
            LOG_ERROR_FMT("[TensorRTInfer] Failed to create TensorRT runtime");
            return false;
        }
        
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
        if (!engine) {
            LOG_ERROR_FMT("[TensorRTInfer] Failed to deserialize CUDA engine");
            delete runtime;
            return false;
        }
        
        // 保存引擎
        engine_.reset(engine);
        
        // 创建执行上下文
        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        if (!context) {
            LOG_ERROR_FMT("[TensorRTInfer] Failed to create execution context");
            return false;
        }
        context_.reset(context);
        
        // 注意：runtime 在引擎创建后可以销毁，但为了简化，这里让它泄漏
        // 或者可以保存起来
        delete runtime;
        
        // 设置类别名称
        class_names_ = {"person", "car", "dog", "bicycle", "motorcycle"};
        
        LOG_INFO_FMT("[TensorRTInfer] Engine loaded successfully from {}", engine_path);
        return true;
        
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[TensorRTInfer] Exception during engine initialization: {}", e.what());
        return false;
    }
}

// 注册节点
REGISTER_NODE("tensorrt_infer", TensorRTInferNode)

} // namespace nodes
} // namespace ai_stream