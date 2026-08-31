// src/nodes/infer/rknn_detection_infer.cpp
// RKNN 检测推理节点——使用 HAL 抽象接口
#include "rknn_detection_infer.h"
#include "ai_stream/core/packet.h"
#include "ai_stream/hal/inference_engine_factory.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

namespace ai_stream {
namespace nodes {

RknnDetectionInferNode::RknnDetectionInferNode() : IInferNode("RknnDetectionInfer") {
    LOG_DEBUG_FMT("[RknnDetectionInfer] Constructor");
}

RknnDetectionInferNode::~RknnDetectionInferNode() {
    stop();
    LOG_DEBUG_FMT("[RknnDetectionInfer] Destructor");
}

bool RknnDetectionInferNode::loadModel(const std::string& model_path) {
    LOG_INFO_FMT("[RknnDetectionInfer] Loading model from: {}", model_path);

    // 通过工厂创建推理引擎（自动选择可用后端）
    engine_ = hal::InferenceEngineFactory::instance().create(backend_type_);
    if (!engine_) {
        LOG_ERROR("[RknnDetectionInfer] Failed to create inference engine");
        return false;
    }

    hal::InferenceConfig config;
    config.model_path = model_path;
    config.input_width = input_width_;
    config.input_height = input_height_;
    config.batch_size = batch_size_;
    config.precision = precision_;
    config.device_id = device_id_;

    if (!engine_->loadModel(config)) {
        LOG_ERROR_FMT("[RknnDetectionInfer] Failed to load model: {}", model_path);
        engine_.reset();
        return false;
    }

    LOG_INFO_FMT("[RknnDetectionInfer] Model loaded via backend: {}", engine_->getBackendName());
    return true;
}

void RknnDetectionInferNode::setPrecision(const std::string& precision) {
    precision_ = precision;
    LOG_INFO_FMT("[RknnDetectionInfer] Set precision: {}", precision);
}

void RknnDetectionInferNode::setBatchSize(int batch_size) {
    batch_size_ = batch_size;
    max_batch_size_ = batch_size;
    queue_.setMaxSize(batch_size_ * 4);
    LOG_INFO_FMT("[RknnDetectionInfer] Set batch size: {}, queue size: {}", batch_size, queue_.getMaxSize());
}

void RknnDetectionInferNode::setInputSize(int width, int height) {
    input_width_ = width;
    input_height_ = height;
}

std::pair<int, int> RknnDetectionInferNode::getInputSize() const {
    if (engine_) {
        return engine_->getInputSize();
    }
    return {input_width_, input_height_};
}

void RknnDetectionInferNode::setClassNames(const std::vector<std::string>& names) {
    class_names_ = names;
}

std::vector<std::string> RknnDetectionInferNode::getClassNames() const {
    return class_names_;
}

void RknnDetectionInferNode::setDetectorType(DetectorType type) {
    detector_type_ = type;
}

DetectorType RknnDetectionInferNode::getDetectorType() const {
    return detector_type_;
}

bool RknnDetectionInferNode::start() {
    if (running_.load()) return true;

    if (!engine_) {
        LOG_ERROR("[RknnDetectionInfer] Cannot start: no engine loaded");
        return false;
    }

    // 如果之前的 worker 线程还未 join，先 join 它（自停后线程可能还在运行）
    if (worker_.joinable()) {
        worker_.join();
    }

    running_ = true;
    worker_ = std::thread(&RknnDetectionInferNode::inferLoop, this);
    LOG_INFO_FMT("[RknnDetectionInfer] Started (backend: {})", engine_->getBackendName());
    return true;
}

void RknnDetectionInferNode::stop() {
    if (!running_.exchange(false)) return;
    queue_.stop();
    if (worker_.joinable()) {
        worker_.join();
    }
    LOG_INFO("[RknnDetectionInfer] Stopped");
}

void RknnDetectionInferNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (!packet || !running_.load()) return;

    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO("[RknnDetectionInfer] Received stream end");
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 inferLoop 中检查，worker 线程会自然退出
        broadcast(packet);
        return;
    }

    if (packet->type != core::PacketType::DECODED_FRAME) return;
    auto frame = std::dynamic_pointer_cast<core::VideoFramePacket>(packet);
    if (!frame || !frame->mat) return;

    // 非阻塞入队（满了丢弃最旧帧）
    if (!queue_.push(frame, std::chrono::milliseconds(10))) {
        std::shared_ptr<core::VideoFramePacket> popped_frame;
        queue_.pop(popped_frame);
        queue_.push(frame, std::chrono::milliseconds(10));
        recordDropped();
    }
}

void RknnDetectionInferNode::inferLoop() {
    while (running_.load()) {
        std::shared_ptr<core::VideoFramePacket> frame;
        if (!queue_.pop(frame, batch_timeout_ms_)) {
            continue;
        }
        if (!frame || !frame->mat) continue;

        // 预处理
        int input_size = input_width_ * input_height_ * 3;
        std::vector<float> input_buffer(input_size);
        preprocess(*frame->mat, input_buffer.data());

        // 推理
        int output_size = 8400 * 84 * sizeof(float);  // YOLOv8 typical output
        std::vector<float> output_buffer(output_size / sizeof(float));

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!engine_->infer(input_buffer.data(), input_buffer.size() * sizeof(float),
                           output_buffer.data(), output_buffer.size() * sizeof(float))) {
            LOG_ERROR("[RknnDetectionInfer] Inference failed");
            continue;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();

        // 后处理
        auto detections = postprocess(output_buffer.data(), output_buffer.size(), 0.5f);

        // 构造结果包
        auto result = std::make_shared<core::InferenceResultPacket>();
        result->stream_id = frame->stream_id;
        result->source_id = frame->source_id;
        result->timestamp_ms = frame->timestamp_ms;
        result->frame_id = frame->frame_id;
        result->detections = std::move(detections);
        result->cost_ms = duration;

        broadcast(result);
    }
}

void RknnDetectionInferNode::preprocess(const cv::Mat& image, float* buffer) {
    // Resize to input size
    cv::Mat resized;
    cv::resize(image, resized, cv::Size(input_width_, input_height_));

    // Convert to float and normalize to [0, 1]
    cv::Mat float_mat;
    resized.convertTo(float_mat, CV_32FC3, 1.0 / 255.0);

    // Convert HWC to CHW
    int plane_size = input_width_ * input_height_;
    std::vector<cv::Mat> channels(3);
    for (int c = 0; c < 3; ++c) {
        channels[c] = cv::Mat(input_height_, input_width_, CV_32FC1,
                              buffer + c * plane_size);
    }
    cv::split(float_mat, channels);
}

std::vector<core::InferenceResultPacket::BBox> RknnDetectionInferNode::postprocess(
    const float* output, int output_size, float conf_thresh) {

    std::vector<core::InferenceResultPacket::BBox> detections;

    // YOLOv8 输出格式: [1, 84, 8400] 或 [8400, 84]
    // 这里简化处理，实际实现需要解析具体模型输出格式
    const int num_candidates = 8400;
    const int num_classes = 80;
    const int elem_size = num_classes + 4;  // 4 bbox + 80 classes

    if (output_size < num_candidates * elem_size) {
        LOG_ERROR_FMT("[RknnDetectionInfer] Output size mismatch: got {}, expected {}",
                      output_size, num_candidates * elem_size);
        return detections;
    }

    for (int i = 0; i < num_candidates; ++i) {
        // 找到最大类别置信度
        float max_conf = 0.0f;
        int max_class = 0;
        for (int c = 0; c < num_classes; ++c) {
            float conf = output[i * elem_size + 4 + c];
            if (conf > max_conf) {
                max_conf = conf;
                max_class = c;
            }
        }

        if (max_conf < conf_thresh) continue;

        core::InferenceResultPacket::BBox bbox;
        bbox.x = output[i * elem_size + 0];
        bbox.y = output[i * elem_size + 1];
        bbox.w = output[i * elem_size + 2];
        bbox.h = output[i * elem_size + 3];
        bbox.confidence = max_conf;
        bbox.class_id = max_class;
        bbox.class_name = (max_class < static_cast<int>(class_names_.size()))
                         ? class_names_[max_class] : "class_" + std::to_string(max_class);

        detections.push_back(bbox);
    }

    return detections;
}

// 注册节点到工厂
REGISTER_NODE("rknn_detection_infer", RknnDetectionInferNode)

} // namespace nodes
} // namespace ai_stream
