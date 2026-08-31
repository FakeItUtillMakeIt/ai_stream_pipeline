// src/nodes/infer/cuda_pose_infer.cu
#include "cuda_pose_infer.h"
#include "pose_postprocess.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/cuda_check.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <chrono>
#include <cmath>

namespace ai_stream {
namespace nodes {

// ============================================================
// CudaPoseInferNode（GPU 预处理路径）
// 原图上传 GPU 后，由 HAL 后端内部完成 crop+letterbox 融合
// kernel 与推理；本节点负责业务逻辑与关键点解码。
// ============================================================

CudaPoseInferNode::CudaPoseInferNode() : IInferNode("CudaPoseInfer") {
    LOG_INFO_FMT("[CudaPoseInfer] Constructor");
}

CudaPoseInferNode::~CudaPoseInferNode() {
    stop();

    if (d_source_img_) { cudaFree(d_source_img_); d_source_img_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
    engine_.reset();

    LOG_INFO_FMT("[CudaPoseInfer] Destructor");
}

bool CudaPoseInferNode::loadModel(const std::string& model_path) {
    LOG_INFO_FMT("[CudaPoseInfer] Loading model from: {}", model_path);
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR_FMT("[CudaPoseInfer] Model file not found: {}", model_path);
        return false;
    }

    engine_ = hal::PoseEstimationFactory::instance().create(
        hal::PoseEstimationBackend::AUTO);
    if (!engine_) {
        LOG_ERROR("[CudaPoseInfer] No pose estimation backend available "
                  "(check WITH_TENSORRT/WITH_RKNN/WITH_ASCEND build options)");
        return false;
    }

    hal::PoseEstimationConfig cfg;
    cfg.model_path = model_path;
    cfg.input_width = input_width_;
    cfg.input_height = input_height_;
    cfg.max_batch = batch_size_;
    cfg.precision = precision_;
    cfg.device_id = device_id_;

    if (!engine_->loadModel(cfg)) {
        LOG_ERROR_FMT("[CudaPoseInfer] Failed to load model via HAL backend: {}", model_path);
        engine_.reset();
        return false;
    }

    if (stream_ == nullptr) {
        cudaSetDevice(device_id_);
        cudaStreamCreate(&stream_);
    }
    engine_->setCudaStream(stream_);
    return true;
}

void CudaPoseInferNode::setPrecision(const std::string& precision) {
    precision_ = precision;
    LOG_INFO_FMT("[CudaPoseInfer] Set precision: {}", precision);
}

void CudaPoseInferNode::setBatchSize(int batch_size) {
    batch_size_ = batch_size;
    queue_.setMaxSize(batch_size_ * 4);
    LOG_INFO_FMT("[CudaPoseInfer] Set max persons per frame: {}, queue size: {}",
                  batch_size, queue_.getMaxSize());
}

std::pair<int, int> CudaPoseInferNode::getInputSize() const {
    if (engine_) return engine_->getInputSize();
    return {input_width_, input_height_};
}

std::vector<std::string> CudaPoseInferNode::getClassNames() const {
    return class_names_;
}

bool CudaPoseInferNode::start() {
    if (!engine_) {
        LOG_WARN_FMT("[CudaPoseInfer] No model loaded, will use mock inference");
    }
    // 如果之前的 worker 线程还未 join，先 join 它（自停后线程可能还在运行）
    if (worker_.joinable()) {
        worker_.join();
    }
    queue_.reset();
    running_ = true;
    worker_ = std::thread(&CudaPoseInferNode::inferLoop, this);
    LOG_INFO_FMT("[CudaPoseInfer] Started with max_persons={}", batch_size_);
    return true;
}

void CudaPoseInferNode::stop() {
    running_ = false;
    queue_.stop();
    if (worker_.joinable()) {
        worker_.join();
    }
    LOG_INFO_FMT("[CudaPoseInfer] Stopped");
}

void CudaPoseInferNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[CudaPoseInfer] Received stream end");
        // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
        // running_ 会在 inferLoop 中检查，worker 线程会自然退出
        broadcast(packet);
        return;
    }
    if (!running_) return;
    if (packet->type != core::PacketType::META_DATA) return;

    auto infer_pkt = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
    if (infer_pkt) {
        queue_.push(infer_pkt, std::chrono::milliseconds(10));
    }
}

// ============================================================
// 推理主循环
// ============================================================
void CudaPoseInferNode::inferLoop() {
    while (running_) {

        std::shared_ptr<core::InferenceResultPacket> packet;
        if (!queue_.pop(packet, std::chrono::milliseconds(100))) {
            continue;
        }

        in_time_ms_ = utils::TimeUtil::currentTimeMs();
        processFrame(packet);

        // 耗时统计
        packet->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
        LOG_INFO_FMT("[CudaPoseInfer] Frame processed: {} persons, total={}ms",
                     packet->pose_results.size(), packet->cost_ms);

        if (packet->source_frame) {
            packet->cost_time_map = packet->source_frame->cost_time_map;
        }
        packet->cost_time_map.insert({name_, packet->cost_ms});

        // 广播结果
        broadcast(packet);
    }
}

// ============================================================
// 单帧处理核心（GPU 预处理版）
// ============================================================
void CudaPoseInferNode::processFrame(std::shared_ptr<core::InferenceResultPacket> packet) {
    if (!packet || !packet->source_frame || !packet->source_frame->source_mat) {
        LOG_ERROR_FMT("[CudaPoseInfer] Invalid packet or missing source_mat");
        return;
    }

    const cv::Mat& source_mat = *packet->source_frame->source_mat;
    if (source_mat.empty()) {
        LOG_ERROR_FMT("[CudaPoseInfer] Source mat is empty");
        return;
    }

    // 1. 收集该帧所有 person 检测框
    std::vector<int> person_indices;
    for (int i = 0; i < static_cast<int>(packet->detections.size()); ++i) {
        const auto& det = packet->detections[i];
        if (det.class_id == person_class_id_ || det.class_name == "person") {
            person_indices.push_back(i);
        }
    }

    int num_persons = static_cast<int>(person_indices.size());
    if (num_persons == 0) {
        LOG_INFO_FMT("[CudaPoseInfer] No person detected in this frame");
        return;
    }

    // 2. 限制上限
    if (num_persons > batch_size_) {
        LOG_WARN_FMT("[CudaPoseInfer] Frame has {} persons, exceeds max_batch {}, truncating",
                     num_persons, batch_size_);
        num_persons = batch_size_;
        person_indices.resize(num_persons);
    }

    // 3. 上传原图到 GPU（兼容 OpenCV 非连续 Mat / ROI）
    int src_w = source_mat.cols;
    int src_h = source_mat.rows;
    size_t src_pitch = static_cast<size_t>(src_w) * 3;
    ensureSourceBuffer(src_w, src_h);

    if (source_mat.isContinuous()) {
        CUDA_CHECK(cudaMemcpyAsync(d_source_img_, source_mat.data,
                                   src_h * src_pitch,
                                   cudaMemcpyHostToDevice, stream_));
    } else {
        CUDA_CHECK(cudaMemcpy2DAsync(
            d_source_img_, src_pitch,
            source_mat.data, source_mat.step,
            src_w * 3, src_h,
            cudaMemcpyHostToDevice, stream_));
    }

    // 4. 计算 letterbox 参数并组装 boxes 参数
    //    boxes格式: [x1, y1, x2, y2, scale, pad_x, pad_y] x num_persons
    const auto [input_w, input_h] = engine_->getInputSize();
    std::vector<float> boxes_7(static_cast<size_t>(num_persons) * 7);
    std::vector<float> h_letterbox_params(static_cast<size_t>(num_persons) * 3);
    for (int i = 0; i < num_persons; ++i) {
        const auto& det = packet->detections[person_indices[i]];
        float crop_w = det.w;
        float crop_h = det.h;

        float scale = fminf(input_w / crop_w, input_h / crop_h);
        float scaled_w = crop_w * scale;
        float scaled_h = crop_h * scale;
        float pad_x = (input_w - scaled_w) / 2.0f;
        float pad_y = (input_h - scaled_h) / 2.0f;

        boxes_7[i * 7 + 0] = det.x;
        boxes_7[i * 7 + 1] = det.y;
        boxes_7[i * 7 + 2] = det.x + det.w;
        boxes_7[i * 7 + 3] = det.y + det.h;
        boxes_7[i * 7 + 4] = scale;
        boxes_7[i * 7 + 5] = pad_x;
        boxes_7[i * 7 + 6] = pad_y;

        h_letterbox_params[i * 3 + 0] = scale;
        h_letterbox_params[i * 3 + 1] = pad_x;
        h_letterbox_params[i * 3 + 2] = pad_y;
    }

    // 5. 通过 HAL 引擎端到端推理（GPU 预处理 kernel 在后端内执行）
    std::vector<float> output_host;
    if (!engine_->inferFromDeviceImage(d_source_img_, src_w, src_h, src_pitch,
                                       boxes_7, num_persons, output_host)) {
        LOG_ERROR_FMT("[CudaPoseInfer] Engine inference failed for batch={}", num_persons);
        return;
    }

    // 6. 后处理：解码关键点
    pose_postprocess::decodeFrame(packet, person_indices, num_persons,
                                  output_host.data(), h_letterbox_params,
                                  conf_thresh_, kpt_conf_thresh_, "CudaPoseInfer");
}

// ============================================================
// 确保原图 GPU 缓冲区足够
// ============================================================
void CudaPoseInferNode::ensureSourceBuffer(int w, int h) {
    size_t needed = static_cast<size_t>(w) * h * 3;
    if (source_buffer_size_ < needed) {
        if (d_source_img_) cudaFree(d_source_img_);
        CUDA_CHECK(cudaMalloc(&d_source_img_, needed));
        source_buffer_size_ = needed;
        LOG_INFO_FMT("[CudaPoseInfer] Reallocated source GPU buffer: {} bytes", needed);
    }
}

REGISTER_NODE("cuda_pose_infer", CudaPoseInferNode)

} // namespace nodes
} // namespace ai_stream
