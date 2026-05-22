// src/nodes/infer/pose_infer.cpp
#include "pose_infer.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "tensor_rt_logger.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>

#include <NvInfer.h>

namespace ai_stream {
namespace nodes {

// ============================================================
// PoseInferNode
// ============================================================

PoseInferNode::PoseInferNode() : IInferNode("PoseInfer") {
    LOG_INFO_FMT("[PoseInfer] Constructor");
}

PoseInferNode::~PoseInferNode() {
    stop();

    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_) cudaStreamDestroy(stream_);

    LOG_INFO_FMT("[PoseInfer] Destructor");
}

bool PoseInferNode::loadModel(const std::string& model_path) {
    LOG_INFO_FMT("[PoseInfer] Loading model from: {}", model_path);
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR_FMT("[PoseInfer] Model file not found: {}", model_path);
        return false;
    }
    return initEngine(model_path);
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
    return {input_width_, input_height_};
}

std::vector<std::string> PoseInferNode::getClassNames() const {
    return class_names_;
}

bool PoseInferNode::start() {
    if (!engine_) {
        LOG_WARN_FMT("[PoseInfer] No model loaded, will use mock inference");
    }
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
    int hw = INPUT_H * INPUT_W;
    int person_stride = 3 * hw;
    alignas(64) static thread_local std::vector<float> host_input;
    host_input.resize(static_cast<size_t>(num_persons) * person_stride);

    // 4. 逐人 crop + preprocess
    for (int i = 0; i < num_persons; ++i) {
        int det_idx = person_indices[i];
        const auto& det = packet->detections[det_idx];

        if (!cropAndPreprocess(source_mat, det, host_input.data(), i)) {
            LOG_WARN_FMT("[PoseInfer] Failed to crop person {}", det_idx);
        }
    }

    // 5. 设置动态 batch 并拷贝到 GPU
    nvinfer1::Dims4 input_dims(num_persons, 3, INPUT_H, INPUT_W);
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
        LOG_ERROR_FMT("[PoseInfer] setInputShape failed for batch={}", num_persons);
        return;
    }

    size_t input_bytes = static_cast<size_t>(num_persons) * person_stride * sizeof(float);
    cudaMemcpyAsync(d_input_, host_input.data(), input_bytes,
                    cudaMemcpyHostToDevice, stream_);

    // 6. 设置输出缓冲区大小（动态 batch 后输出大小会变）
    size_t output_bytes = static_cast<size_t>(num_persons) * NUM_CANDIDATES * POSE_DIM * sizeof(float);

    // 7. 设置 Tensor 地址
    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);

    // 8. 执行推理
    auto t0 = std::chrono::high_resolution_clock::now();
    if (!context_->enqueueV3(stream_)) {
        LOG_ERROR_FMT("[PoseInfer] enqueueV3 failed for batch={}", num_persons);
        return;
    }
    cudaStreamSynchronize(stream_);
    auto t1 = std::chrono::high_resolution_clock::now();

    float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    LOG_INFO_FMT("[PoseInfer] TensorRT enqueue: {} persons, {:.2f}ms", num_persons, infer_ms);

    // 9. 拷贝输出回 host
    h_output_.resize(num_persons * NUM_CANDIDATES * POSE_DIM);
    cudaMemcpyAsync(h_output_.data(), d_output_, output_bytes,
                    cudaMemcpyDeviceToHost, stream_);
    cudaStreamSynchronize(stream_);

    // 10. 后处理：解码关键点
    postprocessFrame(packet, person_indices, num_persons, h_output_.data());
}

// ============================================================
// Crop + Preprocess：从原图按检测框 crop，resize 640，normalize
// ============================================================
bool PoseInferNode::cropAndPreprocess(
    const cv::Mat& source_mat,
    const core::InferenceResultPacket::BBox& det,
    float* host_buffer,
    int slot_idx) {

    // 检测框转 cv::Rect（float 精度）
    float x1 = det.x;
    float y1 = det.y;
    float x2 = det.x + det.w;
    float y2 = det.y + det.h;

    // 边界检查
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

    // Crop
    cv::Rect roi(ix1, iy1, ix2 - ix1, iy2 - iy1);
    cv::Mat cropped = source_mat(roi).clone();

    // Resize to 640x640
    cv::Mat resized;
    cv::resize(cropped, resized, cv::Size(INPUT_W, INPUT_H));

    // Convert to float32 and normalize [0,1]
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32FC3, 1.0 / 255.0);

    // 分离通道并拷贝到 host_buffer
    int hw = INPUT_H * INPUT_W;
    int batch_stride = 3 * hw;
    float* batch_ptr = host_buffer + slot_idx * batch_stride;

    std::vector<cv::Mat> chs(3);
    cv::split(float_img, chs);
    memcpy(batch_ptr + 0 * hw, chs[0].data, hw * sizeof(float));
    memcpy(batch_ptr + 1 * hw, chs[1].data, hw * sizeof(float));
    memcpy(batch_ptr + 2 * hw, chs[2].data, hw * sizeof(float));

    return true;
}

// ============================================================
// 后处理：每人体框独立解码关键点
// 输出格式: [num_persons, 8400, 56] (已转置)
// ============================================================
void PoseInferNode::postprocessFrame(
    std::shared_ptr<core::InferenceResultPacket> packet,
    const std::vector<int>& person_indices,
    int num_persons,
    float* output_host) {

    packet->pose_results.clear();
    packet->pose_results.reserve(num_persons);

    for (int p = 0; p < num_persons; ++p) {
        int det_idx = person_indices[p];
        auto& det = packet->detections[det_idx];

        // 该人的输出起始地址: [p, 0, 0]
        float* person_output = output_host + p * NUM_CANDIDATES * POSE_DIM;

        // 遍历 8400 候选，找 person score 最高的
        float best_score = -1.0f;
        int best_idx = 0;

        for (int i = 0; i < NUM_CANDIDATES; ++i) {
            float score = person_output[i * POSE_DIM + 4];  // 第 5 维是 person score
            // decoded 模式下 score 可能已经是 sigmoid 后的，也可能不是
            // 这里统一做 sigmoid（如果已经是 sigmoid，二次 sigmoid 影响极小）
            score = 1.0f / (1.0f + std::exp(-score));
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        if (best_score < conf_thresh_) {
            LOG_INFO_FMT("[PoseInfer] Person {} best score {:.3f} below threshold {}, skipping",
                          det_idx, best_score, conf_thresh_);
            continue;
        }

        // 取最优候选的 56 维数据
        float* best_pred = person_output + best_idx * POSE_DIM;

        // 解码 bbox (cx, cy, w, h) —— decoded 模式，已是 0~640 像素坐标
        float cx = best_pred[0];
        float cy = best_pred[1];
        float w = best_pred[2];
        float h = best_pred[3];

        // 解码关键点 [51] -> [17, 3]
        core::InferenceResultPacket::PoseResult pose;
        pose.person_score = best_score;
        pose.matched_det_idx = det_idx;

        // 关键点映射比例：检测框尺寸 / 640
        float scale_x = det.w / static_cast<float>(INPUT_W);
        float scale_y = det.h / static_cast<float>(INPUT_H);

        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            float kx = best_pred[5 + k * 3 + 0];
            float ky = best_pred[5 + k * 3 + 1];
            float kconf = best_pred[5 + k * 3 + 2];

            // visibility sigmoid
            kconf = 1.0f / (1.0f + std::exp(-kconf));

            // 映射到原图坐标
            float orig_kx = kx * scale_x + det.x;
            float orig_ky = ky * scale_y + det.y;

            pose.keypoints[k] = {
                orig_kx,
                orig_ky,
                kconf,
                kconf > kpt_conf_thresh_
            };
            // 放到bbox结果中
            det.has_keypoints = true;
            det.keypoints[k]= pose.keypoints[k];
            det.keypoints_conf = kconf;
        }

        // person_box 使用检测框（原图坐标）
        pose.person_box = cv::Rect2f(det.x, det.y, det.w, det.h);

        packet->pose_results.push_back(pose);

        LOG_INFO_FMT("[PoseInfer] Person {}: score={:.3f}, kpts_visible={}/17",
                      det_idx, best_score,
                      std::count_if(pose.keypoints.begin(), pose.keypoints.end(),
                                    [](const auto& k){ return k.visible; }));
    }
}

// ============================================================
// 初始化引擎
// ============================================================
bool PoseInferNode::initEngine(const std::string& engine_path) {
    try {
        // 1. 读取文件
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR_FMT("[PoseInfer] Failed to open engine: {}", engine_path);
            return false;
        }

        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            LOG_ERROR_FMT("[PoseInfer] Failed to read engine");
            return false;
        }

        // 2. 创建 runtime
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
        if (!runtime) {
            LOG_ERROR_FMT("[PoseInfer] createInferRuntime failed");
            return false;
        }

        // 3. 反序列化引擎
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
        if (!engine) {
            LOG_ERROR_FMT("[PoseInfer] deserializeCudaEngine failed");
            delete runtime;
            return false;
        }

        // 4. 创建上下文
        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        if (!context) {
            LOG_ERROR_FMT("[PoseInfer] createExecutionContext failed");
            delete engine;
            delete runtime;
            return false;
        }

        // 5. 保存资源
        runtime_.reset(runtime);
        engine_.reset(engine);
        context_.reset(context);

        // 6. 创建 CUDA Stream
        cudaStreamCreate(&stream_);

        // 打印 Tensor 信息
        int nb_io = engine_->getNbIOTensors();
        LOG_INFO_FMT("[PoseInfer] Engine has {} I/O tensors", nb_io);
        for (int i = 0; i < nb_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
            LOG_INFO_FMT("[PoseInfer]  Tensor[{}]: {} ({}), shape=[{}]",
                         i, name, mode_str, dims.nbDims);
        }

        // 7. 计算缓冲区大小
        int max_batch = batch_size_;
        input_size_ = static_cast<size_t>(max_batch) * 3 * INPUT_H * INPUT_W * sizeof(float);
        output_size_ = static_cast<size_t>(max_batch) * NUM_CANDIDATES * POSE_DIM * sizeof(float);

        // 8. 分配 GPU 内存
        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_output_, output_size_);

        // 9. 预分配 CPU 输出缓冲区
        h_output_.resize(max_batch * NUM_CANDIDATES * POSE_DIM);

        LOG_INFO_FMT("[PoseInfer] Engine loaded: {} (max_persons={}, input={}x{}, candidates={})",
                     engine_path, max_batch, INPUT_W, INPUT_H, NUM_CANDIDATES);
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[PoseInfer] initEngine exception: {}", e.what());
        return false;
    }
}

REGISTER_NODE("pose_infer", PoseInferNode)

} // namespace nodes
} // namespace ai_stream