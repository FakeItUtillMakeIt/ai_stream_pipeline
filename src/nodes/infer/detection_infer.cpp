// src/nodes/infer/detection_infer.cpp
// 【加速优化】Pinned Memory + CUDA Graph + 双流异步传输
#include "detection_infer.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "tensor_rt_logger.h"
#include "utils/cuda_check.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

#include <NvInfer.h>

namespace ai_stream {
namespace nodes {

// ============================================================
// DetectionInferNode - 加速优化版
// ============================================================

DetectionInferNode::DetectionInferNode() : IInferNode("DetectionInfer") {
    LOG_DEBUG_FMT("[DetectionInfer] Constructor");
}

DetectionInferNode::~DetectionInferNode() {
    stop();

    // 释放 CUDA Graph
    destroyCudaGraph();

    // 释放 Pinned Memory
    freePinnedMemory();

    // 释放 GPU 缓冲区
    if (d_input_) cudaFree(d_input_);
    if (d_boxes_) cudaFree(d_boxes_);
    if (d_scores_) cudaFree(d_scores_);
    if (d_classes_) cudaFree(d_classes_);
    if (d_batch_ids_) cudaFree(d_batch_ids_);
    if (d_num_dets_) cudaFree(d_num_dets_);
    if (d_preprocess_tmp_) cudaFree(d_preprocess_tmp_);

    // 释放 CUDA streams
    if (compute_stream_) cudaStreamDestroy(compute_stream_);
    if (transfer_stream_) cudaStreamDestroy(transfer_stream_);

    LOG_DEBUG_FMT("[DetectionInfer] Destructor");
}

bool DetectionInferNode::loadModel(const std::string& model_path) {
    LOG_INFO_FMT("[DetectionInfer] Loading model from: {}", model_path);
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR_FMT("[DetectionInfer] Model file not found: {}", model_path);
        return false;
    }
    return initEngine(model_path);
}

void DetectionInferNode::setPrecision(const std::string& precision) {
    precision_ = precision;
    LOG_INFO_FMT("[DetectionInfer] Set precision: {}", precision);
}

void DetectionInferNode::setBatchSize(int batch_size) {
    batch_size_ = batch_size;
    max_batch_size_ = batch_size;
    queue_.setMaxSize(batch_size_ * 4);
    LOG_INFO_FMT("[DetectionInfer] Set batch size: {}, queue size: {}", batch_size, queue_.getMaxSize());
}

std::pair<int, int> DetectionInferNode::getInputSize() const {
    return {input_width_, input_height_};
}

std::vector<std::string> DetectionInferNode::getClassNames() const {
    return class_names_;
}

bool DetectionInferNode::start() {
    if (!engine_) {
        LOG_WARN_FMT("[DetectionInfer] No model loaded, will use mock inference");
    }
    running_ = true;
    worker_ = std::thread(&DetectionInferNode::inferLoop, this);
    LOG_INFO_FMT("[DetectionInfer] Started with max_batch={}, cuda_graph={}, pinned_memory={}",
                 max_batch_size_.load(), cuda_graph_enabled_.load() ? "ON" : "OFF",
                 h_pinned_input_ ? "ON" : "OFF");
    return true;
}

void DetectionInferNode::stop() {
    running_ = false;
    queue_.stop();
    if (worker_.joinable()) {
        worker_.join();
    }
    LOG_INFO_FMT("[DetectionInfer] Stopped");
}

void DetectionInferNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO_FMT("[DetectionInfer] Received stream end");
        stop();
        broadcast(packet);
        return;
    }
    if (!running_) return;
    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::dynamic_pointer_cast<core::VideoFramePacket>(packet);
    if (frame) {
        queue_.push(frame, std::chrono::milliseconds(10));
    }
}

// ============================================================
// 【加速优化】推理主循环 - 支持 CUDA Graph 快速路径
// ============================================================
void DetectionInferNode::inferLoop() {
    while (running_) {
        in_time_ms_ = utils::TimeUtil::currentTimeMs();
        std::vector<std::shared_ptr<core::VideoFramePacket>> batch_frames;
        batch_frames.reserve(max_batch_size_);

        std::shared_ptr<core::VideoFramePacket> first_frame;
        if (!queue_.pop(first_frame, std::chrono::milliseconds(batch_timeout_ms_))) {
            continue;
        }
        batch_frames.push_back(first_frame);

        auto batch_start_time = std::chrono::high_resolution_clock::now();

        while (static_cast<int>(batch_frames.size()) < max_batch_size_) {
            std::shared_ptr<core::VideoFramePacket> frame;
            auto elapsed = std::chrono::high_resolution_clock::now() - batch_start_time;
            auto remaining = batch_timeout_ms_ - std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
            if (remaining.count() <= 0) break;

            if (queue_.pop(frame, remaining)) {
                batch_frames.push_back(frame);
            } else {
                break;
            }
        }

        int actual_batch = static_cast<int>(batch_frames.size());
        LOG_DEBUG_FMT("[DetectionInfer] Batch collected: {}/{}", actual_batch, max_batch_size_.load());

        auto t0 = std::chrono::high_resolution_clock::now();
        auto results = processBatch(batch_frames);
        auto t1 = std::chrono::high_resolution_clock::now();

        float batch_infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        LOG_INFO_FMT("[DetectionInfer] Batch inference: {} frames, total={:.2f}ms, avg={:.2f}ms/frame",
                     actual_batch, batch_infer_ms, batch_infer_ms / actual_batch);

        for (auto& result : results) {
            if (result) {
                result->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
                result->cost_time_map = first_frame->cost_time_map;
                result->cost_time_map.insert({name_, result->cost_ms});
                broadcast(result);
            }
        }
    }
}

// ============================================================
// 【加速优化】多 batch 推理核心 - 双流 + Pinned Memory
// ============================================================
std::vector<std::shared_ptr<core::InferenceResultPacket>> DetectionInferNode::processBatch(
    const std::vector<std::shared_ptr<core::VideoFramePacket>>& frames) {

    int actual_batch = static_cast<int>(frames.size());
    std::vector<std::shared_ptr<core::InferenceResultPacket>> results;
    results.reserve(actual_batch);

    for (int b = 0; b < actual_batch; ++b) {
        auto result = std::make_shared<core::InferenceResultPacket>();
        result->stream_id = frames[b]->stream_id;
        result->source_id = frames[b]->source_id;
        result->timestamp_ms = frames[b]->timestamp_ms;
        result->source_frame = frames[b];
        result->frame_id = frames[b]->frame_id;
        results.push_back(result);
    }

    if (!engine_ || !context_) {
        // Mock 模式
        static int frame_count = 0;
        for (int b = 0; b < actual_batch; ++b) {
            frame_count++;
            for (int i = 0; i < 3; ++i) {
                core::InferenceResultPacket::BBox box;
                box.x = 100 + (frame_count % 200) + i * 50 + b * 10;
                box.y = 100 + ((frame_count / 2) % 200) + i * 30;
                box.w = 150 + (frame_count % 100);
                box.h = 200 + ((frame_count / 3) % 100);
                box.confidence = 0.7f + (frame_count % 30) / 100.0f;
                box.class_id = (frame_count + i) % class_names_.size();
                box.class_name = class_names_[box.class_id];
                results[b]->detections.push_back(box);
            }
        }
        return results;
    }

    try {
        // 分类：GPU 路径 vs CPU 路径
        std::vector<int> gpu_indices;
        std::vector<int> cpu_indices;
        std::vector<void*> d_ptrs;
        std::vector<size_t> d_pitches;

        float scale_x[actual_batch];
        float scale_y[actual_batch];

        for (int b = 0; b < actual_batch; ++b) {
            // 计算缩放比例
            if (frames[b]->source_mat && !frames[b]->source_mat->empty()) {
                scale_x[b] = static_cast<float>(frames[b]->source_mat->cols) / INPUT_W;
                scale_y[b] = static_cast<float>(frames[b]->source_mat->rows) / INPUT_H;
            } else if (frames[b]->mat && !frames[b]->mat->empty()) {
                scale_x[b] = static_cast<float>(frames[b]->mat->cols) / INPUT_W;
                scale_y[b] = static_cast<float>(frames[b]->mat->rows) / INPUT_H;
            } else {
                scale_x[b] = frames[b]->width / static_cast<float>(INPUT_W);
                scale_y[b] = frames[b]->height / static_cast<float>(INPUT_H);
            }

            // 判断数据来源
            if (frames[b]->is_gpu && frames[b]->d_ptr) {
                gpu_indices.push_back(b);
                d_ptrs.push_back(frames[b]->d_ptr);
                d_pitches.push_back(frames[b]->d_pitch);
            } else {
                if (frames[b]->mat && !frames[b]->mat->empty() && frames[b]->mat->type() == CV_32FC3) {
                    cpu_indices.push_back(b);
                } else {
                    LOG_WARN_FMT("[DetectionInfer] Frame[{}] invalid for CPU path", b);
                }
            }
        }

        int gpu_batch = static_cast<int>(gpu_indices.size());
        int cpu_batch = static_cast<int>(cpu_indices.size());
        int valid_batch = gpu_batch + cpu_batch;

        if (valid_batch == 0) {
            LOG_WARN_FMT("[DetectionInfer] No valid frames in batch");
            return results;
        }

        // 设置动态输入形状
        nvinfer1::Dims4 input_dims(valid_batch, 3, INPUT_H, INPUT_W);
        if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
            LOG_ERROR_FMT("[DetectionInfer] setInputShape failed for batch={}", valid_batch);
            return results;
        }

        // ============================================================
        // 【加速优化】GPU 路径 - 使用 transfer_stream_ 异步传输
        // ============================================================
        if (gpu_batch > 0) {
            LOG_DEBUG_FMT("[DetectionInfer] GPU path: {} frames", gpu_batch);

            for (int i = 0; i < gpu_batch; ++i) {
                size_t offset = i * 3 * INPUT_H * INPUT_W * sizeof(float);
                // 【加速】使用 transfer_stream_ 异步 D2D 拷贝
                cudaMemcpyAsync(static_cast<char*>(d_input_) + offset,
                                d_ptrs[i],
                                3 * INPUT_H * INPUT_W * sizeof(float),
                                cudaMemcpyDeviceToDevice, transfer_stream_);
            }
        }

        // ============================================================
        // 【加速优化】CPU 路径 - 使用 Pinned Memory 加速 H2D 传输
        // ============================================================
        if (cpu_batch > 0) {
            LOG_DEBUG_FMT("[DetectionInfer] CPU path: {} frames (pinned memory)", cpu_batch);

            int hw = INPUT_H * INPUT_W;
            int batch_stride = 3 * hw;

            // 【加速】使用 pinned memory 代替 std::vector<float>
            float* host_input = h_pinned_input_
                ? h_pinned_input_
                : static_cast<float*>(malloc(cpu_batch * batch_stride * sizeof(float)));

            for (int i = 0; i < cpu_batch; ++i) {
                int orig_idx = cpu_indices[i];
                const cv::Mat& image = *frames[orig_idx]->mat;
                const float* img_ptr = image.ptr<float>();
                float* batch_ptr = host_input + i * batch_stride;

                // HWC → NCHW
                for (int h = 0; h < INPUT_H; ++h) {
                    for (int w = 0; w < INPUT_W; ++w) {
                        int src_idx = (h * INPUT_W + w) * 3;
                        int dst_idx = h * INPUT_W + w;
                        batch_ptr[0 * hw + dst_idx] = img_ptr[src_idx + 2]; // R
                        batch_ptr[1 * hw + dst_idx] = img_ptr[src_idx + 1]; // G
                        batch_ptr[2 * hw + dst_idx] = img_ptr[src_idx + 0]; // B
                    }
                }
            }

            // 拷贝到 GPU，接在 GPU 路径数据后面
            size_t gpu_offset = gpu_batch * batch_stride * sizeof(float);
            // 【加速】使用 transfer_stream_ 异步 H2D 传输
            cudaMemcpyAsync(static_cast<char*>(d_input_) + gpu_offset,
                            host_input,
                            cpu_batch * batch_stride * sizeof(float),
                            cudaMemcpyHostToDevice, transfer_stream_);

            // 如果不是 pinned memory，需要释放
            if (!h_pinned_input_) {
                free(host_input);
            }
        }

        // 【加速】等待传输完成后再推理
        cudaStreamSynchronize(transfer_stream_);

        // 设置 Tensor 地址并执行推理
        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(boxes_name_.c_str(), d_boxes_);
        context_->setTensorAddress(scores_name_.c_str(), d_scores_);
        context_->setTensorAddress(classes_name_.c_str(), d_classes_);
        context_->setTensorAddress(batch_ids_name_.c_str(), d_batch_ids_);
        context_->setTensorAddress(num_dets_name_.c_str(), d_num_dets_);

        // ============================================================
        // 【加速优化】推理执行 - 优先使用 CUDA Graph
        // ============================================================
        auto t0 = std::chrono::high_resolution_clock::now();

        if (cuda_graph_ready_ && cuda_graph_batch_size_ == valid_batch) {
            // 【加速】CUDA Graph 快速路径 - 消除 kernel launch overhead
            if (!executeCudaGraph()) {
                LOG_WARN_FMT("[DetectionInfer] CUDA Graph execution failed, fallback to normal");
                if (!context_->enqueueV3(compute_stream_)) {
                    LOG_ERROR_FMT("[DetectionInfer] enqueueV3 failed for batch={}", valid_batch);
                    return results;
                }
                cudaStreamSynchronize(compute_stream_);
            }
        } else {
            // 普通推理路径
            if (!context_->enqueueV3(compute_stream_)) {
                LOG_ERROR_FMT("[DetectionInfer] enqueueV3 failed for batch={}", valid_batch);
                return results;
            }
            cudaStreamSynchronize(compute_stream_);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        LOG_DEBUG_FMT("[DetectionInfer] Inference time: {:.2f}ms (batch={})", infer_ms, valid_batch);

        // ============================================================
        // 【加速优化】后处理 - 使用 Pinned Memory 加速 D2H 传输
        // ============================================================

        // 优先使用 pinned buffers
        int64_t* num_dets_ptr = h_pinned_num_dets_ ? h_pinned_num_dets_ : &h_num_dets_;
        float* boxes_ptr = h_pinned_boxes_ ? h_pinned_boxes_ : h_boxes_.data();
        float* scores_ptr = h_pinned_scores_ ? h_pinned_scores_ : h_scores_.data();
        int64_t* classes_ptr = h_pinned_classes_ ? h_pinned_classes_ : h_classes_.data();
        int64_t* batch_ids_ptr = h_pinned_batch_ids_ ? h_pinned_batch_ids_ : h_batch_ids_.data();

        // 【加速】异步 D2H 传输，使用 transfer_stream_
        cudaMemcpyAsync(num_dets_ptr, d_num_dets_, sizeof(int64_t),
                        cudaMemcpyDeviceToHost, transfer_stream_);
        cudaStreamSynchronize(transfer_stream_);  // 需要先知道检测数量

        int total_dets = static_cast<int>(*num_dets_ptr);
        h_num_dets_ = *num_dets_ptr;

        if (total_dets <= 0) {
            LOG_DEBUG_FMT("[DetectionInfer] No detections in batch");
            return results;
        }

        int max_total_dets = max_batch_size_.load() * MAX_DETS;
        if (total_dets > max_total_dets) {
            total_dets = max_total_dets;
        }

        size_t actual_boxes = static_cast<size_t>(total_dets) * 4 * sizeof(float);
        size_t actual_scores = static_cast<size_t>(total_dets) * sizeof(float);
        size_t actual_classes = static_cast<size_t>(total_dets) * sizeof(int64_t);
        size_t actual_batch_ids = static_cast<size_t>(total_dets) * sizeof(int64_t);

        // 【加速】批量异步 D2H 传输
        cudaMemcpyAsync(boxes_ptr, d_boxes_, actual_boxes, cudaMemcpyDeviceToHost, transfer_stream_);
        cudaMemcpyAsync(scores_ptr, d_scores_, actual_scores, cudaMemcpyDeviceToHost, transfer_stream_);
        cudaMemcpyAsync(classes_ptr, d_classes_, actual_classes, cudaMemcpyDeviceToHost, transfer_stream_);
        cudaMemcpyAsync(batch_ids_ptr, d_batch_ids_, actual_batch_ids, cudaMemcpyDeviceToHost, transfer_stream_);
        cudaStreamSynchronize(transfer_stream_);

        // 同步到 h_boxes_ 等（如果使用了 pinned memory）
        if (h_pinned_boxes_) {
            h_boxes_.assign(boxes_ptr, boxes_ptr + total_dets * 4);
            h_scores_.assign(scores_ptr, scores_ptr + total_dets);
            h_classes_.assign(classes_ptr, classes_ptr + total_dets);
            h_batch_ids_.assign(batch_ids_ptr, batch_ids_ptr + total_dets);
        }

        // 构建 scale 数组
        std::vector<float> valid_scale_x(valid_batch);
        std::vector<float> valid_scale_y(valid_batch);
        int idx = 0;
        for (int i : gpu_indices) {
            valid_scale_x[idx] = scale_x[i];
            valid_scale_y[idx] = scale_y[i];
            idx++;
        }
        for (int i : cpu_indices) {
            valid_scale_x[idx] = scale_x[i];
            valid_scale_y[idx] = scale_y[i];
            idx++;
        }

        auto all_detections = postprocessBatch(valid_batch, total_dets,
                                                valid_scale_x.data(), valid_scale_y.data(), 0.25f);

        // 映射回原始帧索引
        idx = 0;
        for (int i : gpu_indices) {
            results[i]->detections = std::move(all_detections[idx++]);
        }
        for (int i : cpu_indices) {
            results[i]->detections = std::move(all_detections[idx++]);
        }

        LOG_INFO_FMT("[DetectionInfer] Batch done: gpu={} cpu={} total_valid={} total={}, detections={}",
                     gpu_batch, cpu_batch, valid_batch, total_dets, results[0]->detections.size());

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] Batch inference exception: {}", e.what());
    }

    return results;
}

// ============================================================
// 【保留】CPU 路径预处理
// ============================================================
void DetectionInferNode::preprocessBatchCpu(const std::vector<cv::Mat*>& images,
                                             float* gpu_buffer, int batch_size) {
    int hw = INPUT_H * INPUT_W;
    int batch_stride = 3 * hw;

    float* host_input = h_pinned_input_
        ? h_pinned_input_
        : static_cast<float*>(malloc(batch_size * batch_stride * sizeof(float)));

    for (int b = 0; b < batch_size; ++b) {
        const cv::Mat& image = *images[b];
        const float* img_ptr = image.ptr<float>();
        float* batch_ptr = host_input + b * batch_stride;

        for (int h = 0; h < INPUT_H; ++h) {
            for (int w = 0; w < INPUT_W; ++w) {
                int src_idx = (h * INPUT_W + w) * 3;
                int dst_idx = h * INPUT_W + w;
                batch_ptr[0 * hw + dst_idx] = img_ptr[src_idx + 2]; // R
                batch_ptr[1 * hw + dst_idx] = img_ptr[src_idx + 1]; // G
                batch_ptr[2 * hw + dst_idx] = img_ptr[src_idx + 0]; // B
            }
        }
    }

    cudaMemcpyAsync(gpu_buffer, host_input,
                    batch_size * batch_stride * sizeof(float),
                    cudaMemcpyHostToDevice, transfer_stream_);

    if (!h_pinned_input_) {
        free(host_input);
    }
}

// ============================================================
// 后处理
// ============================================================
std::vector<std::vector<core::InferenceResultPacket::BBox>> DetectionInferNode::postprocessBatch(
    int batch_size, int total_dets, const float scale_x[], const float scale_y[], float conf_thresh) {

    std::vector<std::vector<core::InferenceResultPacket::BBox>> all_detections(batch_size);

    for (int i = 0; i < total_dets; ++i) {
        float score = h_scores_[i];
        if (score < conf_thresh) continue;

        int batch_id = static_cast<int>(h_batch_ids_[i]);
        if (batch_id < 0 || batch_id >= batch_size) {
            LOG_WARN_FMT("[DetectionInfer] Invalid batch_id {} at det {}, max={}", batch_id, i, batch_size);
            continue;
        }

        float cx = h_boxes_[i * 4 + 0];
        float cy = h_boxes_[i * 4 + 1];
        float w  = h_boxes_[i * 4 + 2];
        float h  = h_boxes_[i * 4 + 3];

        core::InferenceResultPacket::BBox box;
        box.x = static_cast<int>((cx - w / 2.0f) * scale_x[batch_id]);
        box.y = static_cast<int>((cy - h / 2.0f) * scale_y[batch_id]);
        box.w = static_cast<int>(w * scale_x[batch_id]);
        box.h = static_cast<int>(h * scale_y[batch_id]);
        box.confidence = score;
        box.class_id = static_cast<int>(h_classes_[i]);

        if (box.class_id >= 0 && box.class_id < static_cast<int>(class_names_.size())) {
            box.class_name = class_names_[box.class_id];
        } else {
            box.class_name = "unknown";
        }

        all_detections[batch_id].push_back(box);
    }
    return all_detections;
}

// ============================================================
// 【加速优化】CUDA Graph 捕获 - 消除 kernel launch overhead
// ============================================================
bool DetectionInferNode::captureCudaGraph(int batch_size) {
    if (!engine_ || !context_) return false;

    destroyCudaGraph();

    try {
        // 设置输入形状
        nvinfer1::Dims4 input_dims(batch_size, 3, INPUT_H, INPUT_W);
        if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph: setInputShape failed");
            return false;
        }

        // 设置 tensor 地址
        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(boxes_name_.c_str(), d_boxes_);
        context_->setTensorAddress(scores_name_.c_str(), d_scores_);
        context_->setTensorAddress(classes_name_.c_str(), d_classes_);
        context_->setTensorAddress(batch_ids_name_.c_str(), d_batch_ids_);
        context_->setTensorAddress(num_dets_name_.c_str(), d_num_dets_);

        // 开始捕获 CUDA Graph
        cudaStreamBeginCapture(compute_stream_, cudaStreamCaptureModeGlobal);

        // 执行推理（会被捕获到 graph 中）
        if (!context_->enqueueV3(compute_stream_)) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph capture: enqueueV3 failed");
            cudaStreamEndCapture(compute_stream_, &cuda_graph_);
            return false;
        }

        // 结束捕获
        cudaError_t err = cudaStreamEndCapture(compute_stream_, &cuda_graph_);
        if (err != cudaSuccess || !cuda_graph_) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph capture failed: {}", cudaGetErrorString(err));
            return false;
        }

        // 实例化 graph
        err = cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph instantiate failed: {}", cudaGetErrorString(err));
            return false;
        }

        cuda_graph_batch_size_ = batch_size;
        cuda_graph_ready_ = true;

        LOG_INFO_FMT("[DetectionInfer] CUDA Graph captured and instantiated for batch_size={}", batch_size);
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] CUDA Graph capture exception: {}", e.what());
        return false;
    }
}

bool DetectionInferNode::executeCudaGraph() {
    if (!cuda_graph_ready_ || !cuda_graph_exec_) return false;

    cudaError_t err = cudaGraphLaunch(cuda_graph_exec_, compute_stream_);
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("[DetectionInfer] CUDA Graph launch failed: {}", cudaGetErrorString(err));
        return false;
    }

    cudaStreamSynchronize(compute_stream_);
    return true;
}

void DetectionInferNode::destroyCudaGraph() {
    if (cuda_graph_exec_) {
        cudaGraphExecDestroy(cuda_graph_exec_);
        cuda_graph_exec_ = nullptr;
    }
    if (cuda_graph_) {
        cudaGraphDestroy(cuda_graph_);
        cuda_graph_ = nullptr;
    }
    cuda_graph_ready_ = false;
    cuda_graph_batch_size_ = 0;
}

// ============================================================
// 【加速优化】Pinned Memory 管理
// ============================================================
bool DetectionInferNode::allocatePinnedMemory() {
    int max_batch = max_batch_size_.load();
    int hw = INPUT_H * INPUT_W;
    int batch_stride = 3 * hw;

    size_t input_bytes = static_cast<size_t>(max_batch) * batch_stride * sizeof(float);
    size_t boxes_bytes = static_cast<size_t>(max_batch) * MAX_DETS * 4 * sizeof(float);
    size_t scores_bytes = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(float);
    size_t classes_bytes = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);
    size_t batch_ids_bytes = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);

    cudaError_t err;

    // 分配 pinned memory (page-locked)
    err = cudaMallocHost(&h_pinned_input_, input_bytes);
    if (err != cudaSuccess) {
        LOG_WARN_FMT("[DetectionInfer] Failed to allocate pinned input memory, fallback to regular");
        h_pinned_input_ = nullptr;
    }

    err = cudaMallocHost(&h_pinned_boxes_, boxes_bytes);
    if (err != cudaSuccess) {
        LOG_WARN_FMT("[DetectionInfer] Failed to allocate pinned boxes memory");
        h_pinned_boxes_ = nullptr;
    }

    err = cudaMallocHost(&h_pinned_scores_, scores_bytes);
    if (err != cudaSuccess) {
        LOG_WARN_FMT("[DetectionInfer] Failed to allocate pinned scores memory");
        h_pinned_scores_ = nullptr;
    }

    err = cudaMallocHost(&h_pinned_classes_, classes_bytes);
    if (err != cudaSuccess) {
        LOG_WARN_FMT("[DetectionInfer] Failed to allocate pinned classes memory");
        h_pinned_classes_ = nullptr;
    }

    err = cudaMallocHost(&h_pinned_batch_ids_, batch_ids_bytes);
    if (err != cudaSuccess) {
        LOG_WARN_FMT("[DetectionInfer] Failed to allocate pinned batch_ids memory");
        h_pinned_batch_ids_ = nullptr;
    }

    err = cudaMallocHost(&h_pinned_num_dets_, sizeof(int64_t));
    if (err != cudaSuccess) {
        LOG_WARN_FMT("[DetectionInfer] Failed to allocate pinned num_dets memory");
        h_pinned_num_dets_ = nullptr;
    }

    bool success = (h_pinned_input_ != nullptr);
    LOG_INFO_FMT("[DetectionInfer] Pinned memory allocated: {}", success ? "SUCCESS" : "PARTIAL");
    return success;
}

void DetectionInferNode::freePinnedMemory() {
    if (h_pinned_input_) { cudaFreeHost(h_pinned_input_); h_pinned_input_ = nullptr; }
    if (h_pinned_boxes_) { cudaFreeHost(h_pinned_boxes_); h_pinned_boxes_ = nullptr; }
    if (h_pinned_scores_) { cudaFreeHost(h_pinned_scores_); h_pinned_scores_ = nullptr; }
    if (h_pinned_classes_) { cudaFreeHost(h_pinned_classes_); h_pinned_classes_ = nullptr; }
    if (h_pinned_batch_ids_) { cudaFreeHost(h_pinned_batch_ids_); h_pinned_batch_ids_ = nullptr; }
    if (h_pinned_num_dets_) { cudaFreeHost(h_pinned_num_dets_); h_pinned_num_dets_ = nullptr; }
}

// ============================================================
// 初始化引擎（包含加速优化初始化）
// ============================================================
bool DetectionInferNode::initEngine(const std::string& engine_path) {
    try {
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR_FMT("[DetectionInfer] Failed to open engine: {}", engine_path);
            return false;
        }

        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            LOG_ERROR_FMT("[DetectionInfer] Failed to read engine");
            return false;
        }

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
        if (!runtime) {
            LOG_ERROR_FMT("[DetectionInfer] createInferRuntime failed");
            return false;
        }

        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
        if (!engine) {
            LOG_ERROR_FMT("[DetectionInfer] deserializeCudaEngine failed");
            delete runtime;
            return false;
        }

        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        if (!context) {
            LOG_ERROR_FMT("[DetectionInfer] createExecutionContext failed");
            delete engine;
            delete runtime;
            return false;
        }

        runtime_.reset(runtime);
        engine_.reset(engine);
        context_.reset(context);

        // 【加速优化】创建双流架构
        cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking);

        int nb_io = engine_->getNbIOTensors();
        LOG_INFO_FMT("[DetectionInfer] Engine has {} I/O tensors", nb_io);

        for (int i = 0; i < nb_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
            LOG_INFO_FMT("[DetectionInfer]  Tensor[{}]: {} ({}), shape=[{}]",
                         i, name, mode_str, dims.nbDims);
        }

        int max_batch = max_batch_size_.load();
        input_size_ = static_cast<size_t>(max_batch) * 3 * INPUT_H * INPUT_W * sizeof(float);
        out_boxes_size_ = static_cast<size_t>(max_batch) * MAX_DETS * 4 * sizeof(float);
        out_scores_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(float);
        out_classes_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);
        out_batch_ids_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);
        out_num_dets_size_ = sizeof(int64_t);

        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_boxes_, out_boxes_size_);
        cudaMalloc(&d_scores_, out_scores_size_);
        cudaMalloc(&d_classes_, out_classes_size_);
        cudaMalloc(&d_batch_ids_, out_batch_ids_size_);
        cudaMalloc(&d_num_dets_, out_num_dets_size_);

        cudaMalloc(&d_preprocess_tmp_, input_size_);
        d_preprocess_tmp_size_ = input_size_;

        h_boxes_.resize(max_batch * MAX_DETS * 4);
        h_scores_.resize(max_batch * MAX_DETS);
        h_classes_.resize(max_batch * MAX_DETS);
        h_batch_ids_.resize(max_batch * MAX_DETS);
        h_num_dets_ = 0;

        // 【加速优化】分配 Pinned Memory
        allocatePinnedMemory();

        // 【加速优化】预捕获 CUDA Graph
        if (cuda_graph_enabled_) {
            captureCudaGraph(max_batch);
        }

        LOG_INFO_FMT("[DetectionInfer] Engine loaded: {} (max_batch={}, max_dets={}, streams=2, pinned={}, cuda_graph={})",
                     engine_path, max_batch, MAX_DETS,
                     h_pinned_input_ ? "ON" : "OFF",
                     cuda_graph_ready_ ? "READY" : "OFF");
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] initEngine exception: {}", e.what());
        return false;
    }
}

REGISTER_NODE("detection_infer", DetectionInferNode)

} // namespace nodes
} // namespace ai_stream
