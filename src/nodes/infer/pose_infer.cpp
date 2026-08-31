// src/nodes/infer/pose_infer.cpp
#include "pose_infer.h"
#include "pose_postprocess.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <opencv2/opencv.hpp>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>

namespace ai_stream {
namespace nodes {

// ============================================================
// PoseInferNode（host 预处理路径）
// 推理通过 HAL 后端引擎完成，本节点负责业务逻辑：
// person 框收集、CPU crop+letterbox 预处理、关键点解码。
// ============================================================

PoseInferNode::PoseInferNode() : IInferNode("PoseInfer") {
    LOG_INFO_FMT("[PoseInfer] Constructor");
}

PoseInferNode::~PoseInferNode() {
    stop();
    engine_.reset();
    LOG_INFO_FMT("[PoseInfer] Destructor");
}

bool PoseInferNode::loadModel(const std::string& model_path) {
    LOG_INFO_FMT("[PoseInfer] Loading model from: {}", model_path);
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR_FMT("[PoseInfer] Model file not found: {}", model_path);
        return false;
    }

    engine_ = hal::PoseEstimationFactory::instance().create(
        hal::PoseEstimationBackend::AUTO);
    if (!engine_) {
        LOG_ERROR("[PoseInfer] No pose estimation backend available "
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
        LOG_ERROR_FMT("[PoseInfer] Failed to load model via HAL backend: {}", model_path);
        engine_.reset();
        return false;
    }
    return true;
}

void PoseInferNode::setPrecision(const std::string& precision) {
    precision_ = precision;
    LOG_INFO_FMT("[PoseInfer] Set precision: {}", precision);
}

void PoseInferNode::setBatchSize(int batch_size) {
    batch_size_ = batch_size;
    queue_.setMaxSize(batch_size_ * 4);
    LOG_INFO_FMT("[PoseInfer] Set max persons per frame: {}, queue size: {}",
                  batch_size, queue_.getMaxSize());
}

std::pair<int, int> PoseInferNode::getInputSize() const {
    if (engine_) return engine_->getInputSize();
    return {input_width_, input_height_};
}

std::vector<std::string> PoseInferNode::getClassNames() const {
    return class_names_;
}

bool PoseInferNode::start() {
    if (!engine_) {
        LOG_WARN_FMT("[PoseInfer] No model loaded, will use mock inference");
    }
    // 如果之前的 worker 线程还未 join，先 join 它（自停后线程可能还在运行）
    if (worker_.joinable()) {
        worker_.join();
    }
    queue_.reset();
    running_ = true;
    worker_ = std::thread(&PoseInferNode::inferLoop, this);
    LOG_INFO_FMT("[PoseInfer] Started with max_persons={}", batch_size_);
    return true;
}

void PoseInferNode::stop() {
    running_ = false;
    queue_.stop();
    if (worker_.joinable()) {
        worker_.join();
    }
    LOG_INFO_FMT("[PoseInfer] Stopped");
}

void PoseInferNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[PoseInfer] Received stream end");
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
// 推理主循环：来一帧处理一帧
// ============================================================
void PoseInferNode::inferLoop() {
    while (running_) {
        in_time_ms_ = utils::TimeUtil::currentTimeMs();

        std::shared_ptr<core::InferenceResultPacket> packet;
        if (!queue_.pop(packet, std::chrono::milliseconds(100))) {
            continue;
        }

        auto t0 = std::chrono::high_resolution_clock::now();
        processFrame(packet);
        auto t1 = std::chrono::high_resolution_clock::now();

        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        LOG_INFO_FMT("[PoseInfer] Frame processed: {} persons, total={:.2f}ms",
                     packet->pose_results.size(), infer_ms);

        // 耗时统计
        packet->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
        if (packet->source_frame) {
            packet->cost_time_map = packet->source_frame->cost_time_map;
        }
        packet->cost_time_map.insert({name_, packet->cost_ms});

        // 广播结果
        broadcast(packet);
    }
}

// ============================================================
// 单帧处理核心
// ============================================================
void PoseInferNode::processFrame(std::shared_ptr<core::InferenceResultPacket> packet) {
    if (!packet || !packet->source_frame || !packet->source_frame->source_mat) {
        LOG_ERROR_FMT("[PoseInfer] Invalid packet or missing source_mat");
        return;
    }

    const cv::Mat& source_mat = *packet->source_frame->source_mat;
    if (source_mat.empty()) {
        LOG_ERROR_FMT("[PoseInfer] Source mat is empty");
        return;
    }

    // 1. 收集该帧所有 person 检测框的索引
    std::vector<int> person_indices;
    for (int i = 0; i < static_cast<int>(packet->detections.size()); ++i) {
        const auto& det = packet->detections[i];
        if (det.class_id == person_class_id_ || det.class_name == "person") {
            person_indices.push_back(i);
        }
    }

    int num_persons = static_cast<int>(person_indices.size());
    if (num_persons == 0) {
        LOG_INFO_FMT("[PoseInfer] No person detected in this frame");
        return;
    }

    // 2. 限制上限
    if (num_persons > batch_size_) {
        LOG_WARN_FMT("[PoseInfer] Frame has {} persons, exceeds max_batch {}, truncating",
                     num_persons, batch_size_);
        num_persons = batch_size_;
        person_indices.resize(num_persons);
    }

    // 3. 分配 host 输入缓冲区 [num_persons, 3, 640, 640]
    const auto [input_w, input_h] = engine_->getInputSize();
    int hw = input_h * input_w;
    int person_stride = 3 * hw;
    alignas(64) static thread_local std::vector<float> host_input;
    host_input.resize(static_cast<size_t>(num_persons) * person_stride);

    // 4. 逐人 crop + letterbox preprocess
    std::vector<float> h_letterbox_params(static_cast<size_t>(num_persons) * 3);
    for (int i = 0; i < num_persons; ++i) {
        int det_idx = person_indices[i];
        const auto& det = packet->detections[det_idx];

        float scale, pad_x, pad_y;
        if (!cropAndPreprocess(source_mat, det, host_input.data(), i,
                               scale, pad_x, pad_y)) {
            LOG_WARN_FMT("[PoseInfer] Failed to crop person {}", det_idx);
        }
        h_letterbox_params[i * 3 + 0] = scale;
        h_letterbox_params[i * 3 + 1] = pad_x;
        h_letterbox_params[i * 3 + 2] = pad_y;
    }

    // 5. 通过 HAL 引擎推理
    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> output_host;
    if (!engine_->inferHost(host_input.data(), num_persons, output_host)) {
        LOG_ERROR_FMT("[PoseInfer] Engine inference failed for batch={}", num_persons);
        return;
    }
    auto t1 = std::chrono::high_resolution_clock::now();

    float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    LOG_INFO_FMT("[PoseInfer] Backend inference: {} persons, {:.2f}ms", num_persons, infer_ms);

    // 6. 后处理：解码关键点
    pose_postprocess::decodeFrame(packet, person_indices, num_persons,
                                  output_host.data(), h_letterbox_params,
                                  conf_thresh_, kpt_conf_thresh_, "PoseInfer");
}

// ============================================================
// Crop + Letterbox Preprocess：从原图按检测框 crop，等比缩放+灰色padding，normalize
// ============================================================
bool PoseInferNode::cropAndPreprocess(
    const cv::Mat& source_mat,
    const core::InferenceResultPacket::BBox& det,
    float* host_buffer,
    int slot_idx,
    float& out_scale,
    float& out_pad_x,
    float& out_pad_y) {

    const auto [input_w, input_h] = engine_->getInputSize();

    float x1 = det.x;
    float y1 = det.y;
    float x2 = det.x + det.w;
    float y2 = det.y + det.h;

    int img_w = source_mat.cols;
    int img_h = source_mat.rows;
    int ix1 = std::max(0, static_cast<int>(x1));
    int iy1 = std::max(0, static_cast<int>(y1));
    int ix2 = std::min(img_w, static_cast<int>(x2));
    int iy2 = std::min(img_h, static_cast<int>(y2));

    if (ix1 >= ix2 || iy1 >= iy2) {
        LOG_WARN_FMT("[PoseInfer] Invalid crop box: ({},{})->({},{}) for image {}x{}",
                     ix1, iy1, ix2, iy2, img_w, img_h);
        return false;
    }

    cv::Rect roi(ix1, iy1, ix2 - ix1, iy2 - iy1);
    cv::Mat cropped = source_mat(roi).clone();

    float crop_w = static_cast<float>(cropped.cols);
    float crop_h = static_cast<float>(cropped.rows);

    out_scale = std::min(static_cast<float>(input_w) / crop_w,
                          static_cast<float>(input_h) / crop_h);
    float scaled_w = crop_w * out_scale;
    float scaled_h = crop_h * out_scale;
    out_pad_x = (input_w - scaled_w) / 2.0f;
    out_pad_y = (input_h - scaled_h) / 2.0f;

    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(static_cast<int>(scaled_w), static_cast<int>(scaled_h)));

    cv::Mat letterbox(input_h, input_w, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(letterbox(cv::Rect(static_cast<int>(out_pad_x), static_cast<int>(out_pad_y),
                                      resized.cols, resized.rows)));

    cv::Mat float_img;
    letterbox.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    int hw = input_h * input_w;
    int batch_stride = 3 * hw;
    float* batch_ptr = host_buffer + slot_idx * batch_stride;

    std::vector<cv::Mat> chs(3);
    cv::split(float_img, chs);
    memcpy(batch_ptr + 0 * hw, chs[0].data, hw * sizeof(float));
    memcpy(batch_ptr + 1 * hw, chs[1].data, hw * sizeof(float));
    memcpy(batch_ptr + 2 * hw, chs[2].data, hw * sizeof(float));

    return true;
}

REGISTER_NODE("pose_infer", PoseInferNode)

} // namespace nodes
} // namespace ai_stream
