// src/nodes/infer/detection_infer.cpp
// 检测推理节点——使用 HAL 抽象接口
// 【加速优化】Pinned Memory + CUDA Graph + 双流异步传输
#include "detection_infer.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#ifdef WITH_CUDA
#include "utils/cuda_check.h"
#endif
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <memory>

#ifdef WITH_TENSORRT
#include <NvInfer.h>
#endif

namespace ai_stream {
namespace nodes {

// ============================================================
// DetectionInferNode - HAL 加速优化版
// ============================================================

DetectionInferNode::DetectionInferNode() : core::QueuedNode<IInferNode>("DetectionInfer") {
    LOG_DEBUG_FMT("[DetectionInfer] Constructor");
}

DetectionInferNode::~DetectionInferNode() {
    stop();

    // 释放 CUDA Graph
    destroyCudaGraph();

    // 先同步所有 CUDA 流，确保 GPU 操作完成后再释放资源
    if (compute_stream_) cudaStreamSynchronize(compute_stream_);
    if (transfer_stream_) cudaStreamSynchronize(transfer_stream_);

    // 先销毁推理引擎（引擎内部会调用 cudaFree）
    engine_.reset();

    // 释放 Pinned Memory
    freePinnedMemory();

    // 释放 GPU 缓冲区
    if (d_input_) cudaFree(d_input_);
    if (d_boxes_) cudaFree(d_boxes_);
    if (d_scores_) cudaFree(d_scores_);
    if (d_classes_) cudaFree(d_classes_);
    if (d_batch_ids_) cudaFree(d_batch_ids_);
    if (d_num_dets_) cudaFree(d_num_dets_);

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
    setQueueCapacity(static_cast<size_t>(batch_size > 0 ? batch_size * 4 : 64));
    LOG_INFO_FMT("[DetectionInfer] Set batch size: {}", batch_size);
}

std::pair<int, int> DetectionInferNode::getInputSize() const {
    if (engine_) {
        return engine_->getInputSize();
    }
    return {input_width_, input_height_};
}

std::vector<std::string> DetectionInferNode::getClassNames() const {
    return class_names_;
}

bool DetectionInferNode::onStartup() {
    if (!engine_) {
        LOG_WARN_FMT("[DetectionInfer] No model loaded, will use mock inference");
    }
    // 批次超时 flush 依赖 onIdle，把空闲轮询间隔设为批次窗口
    setPollTimeout(batch_timeout_ms_);
    LOG_INFO_FMT("[DetectionInfer] Started with max_batch={}, backend={}, cuda_graph={}, pinned_memory={}",
                 max_batch_size_.load(),
                 engine_ ? engine_->getBackendName() : "none",
                 cuda_graph_enabled_.load() ? "ON" : "OFF",
                 h_pinned_input_ ? "ON" : "OFF");
    return true;
}

void DetectionInferNode::onShutdown() {
    LOG_INFO_FMT("[DetectionInfer] Stopped");
}

void DetectionInferNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (!packet) return;

    if (packet->type == core::PacketType::STREAM_END) {
        // 先处理完在途批次，再转发 STREAM_END（基类随后统一 stop）
        flushBatch();
        LOG_INFO_FMT("[DetectionInfer] Stream end");
        broadcast(packet);
        return;
    }
    if (packet->type != core::PacketType::DECODED_FRAME) {
        return;
    }

    auto frame = std::static_pointer_cast<core::VideoFramePacket>(packet);
    if (!batch_active_) {
        batch_active_ = true;
        batch_start_tp_ = std::chrono::steady_clock::now();
        batch_start_ms_ = utils::TimeUtil::currentTimeMs();
    }
    batch_frames_.push_back(std::move(frame));
    if (static_cast<int>(batch_frames_.size()) >= max_batch_size_.load()) {
        flushBatch();
    }
}

void DetectionInferNode::onIdle() {
    if (!batch_active_) return;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - batch_start_tp_).count();
    if (elapsed >= batch_timeout_ms_.count()) {
        flushBatch();
    }
}

// ============================================================
// 批次处理 - 支持 CUDA Graph 快速路径
// ============================================================
void DetectionInferNode::flushBatch() {
    if (batch_frames_.empty()) {
        batch_active_ = false;
        return;
    }
    auto batch_frames = std::move(batch_frames_);
    batch_frames_.clear();
    batch_active_ = false;

    int actual_batch = static_cast<int>(batch_frames.size());
    LOG_DEBUG_FMT("[DetectionInfer] Batch collected: {}/{}", actual_batch, max_batch_size_.load());

    auto t0 = std::chrono::high_resolution_clock::now();
    auto results = processBatch(batch_frames);
    auto t1 = std::chrono::high_resolution_clock::now();

    float batch_infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    LOG_INFO_FMT("[DetectionInfer] Batch inference: {} frames, total={:.2f}ms, avg={:.2f}ms/frame",
                 actual_batch, batch_infer_ms, batch_infer_ms / actual_batch);

    const auto& frame_cost_map = batch_frames.front()->cost_time_map;
    uint64_t now_ms = utils::TimeUtil::currentTimeMs();
    for (auto& result : results) {
        if (result) {
            result->cost_ms = now_ms - batch_start_ms_;
            result->cost_time_map = frame_cost_map;
            result->cost_time_map.insert({name_, result->cost_ms});
            broadcast(result);
        }
    }
}

// ============================================================
// 多 batch 推理核心 - 双流 + Pinned Memory
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

    if (!engine_) {
        // Mock 模式
        for (int b = 0; b < actual_batch; ++b) {
            int frame_count = mock_frame_count_.fetch_add(1);
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

        std::vector<float> scale_x(actual_batch);
        std::vector<float> scale_y(actual_batch);

        for (int b = 0; b < actual_batch; ++b) {
            // 计算缩放比例
            if (frames[b]->source_mat && !frames[b]->source_mat->empty()) {
                scale_x[b] = static_cast<float>(frames[b]->source_mat->cols) / input_width_;
                scale_y[b] = static_cast<float>(frames[b]->source_mat->rows) / input_height_;
            } else if (frames[b]->mat && !frames[b]->mat->empty()) {
                scale_x[b] = static_cast<float>(frames[b]->mat->cols) / input_width_;
                scale_y[b] = static_cast<float>(frames[b]->mat->rows) / input_height_;
            } else {
                scale_x[b] = frames[b]->width / static_cast<float>(input_width_);
                scale_y[b] = frames[b]->height / static_cast<float>(input_height_);
            }

            // 判断数据来源
            if (frames[b]->is_gpu && frames[b]->d_ptr) {
                gpu_indices.push_back(b);
                d_ptrs.push_back(frames[b]->d_ptr);
                d_pitches.push_back(frames[b]->d_pitch);
            } else {
                const bool nv12_ok = input_nv12_ && frames[b]->is_nv12 && frames[b]->nv12;
                if (nv12_ok ||
                    (frames[b]->mat && !frames[b]->mat->empty() && frames[b]->mat->type() == CV_32FC3)) {
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


#ifdef WITH_CUDA
#ifdef WITH_TENSORRT
        // 设置动态输入形状（通过原始 TensorRT context，仅 TensorRT 后端）
        if (auto* raw_context = graph_engine_ ? static_cast<nvinfer1::IExecutionContext*>(graph_engine_->getRawContext()) : nullptr) {
            nvinfer1::Dims4 input_dims(valid_batch, 3, input_height_, input_width_);
            if (!raw_context->setInputShape(input_name_.c_str(), input_dims)) {
                LOG_ERROR_FMT("[DetectionInfer] setInputShape failed for batch={}", valid_batch);
                return results;
            }
        }
#endif

        // ============================================================
        // GPU 路径 - 使用 transfer_stream_ 异步传输
        // ============================================================
        if (gpu_batch > 0) {
            LOG_DEBUG_FMT("[DetectionInfer] GPU path: {} frames", gpu_batch);

            for (int i = 0; i < gpu_batch; ++i) {
                size_t offset = i * 3 * input_height_ * input_width_ * sizeof(float);
                cudaMemcpyAsync(static_cast<char*>(d_input_) + offset,
                                d_ptrs[i],
                                3 * input_height_ * input_width_ * sizeof(float),
                                cudaMemcpyDeviceToDevice, transfer_stream_);
            }
        }

        // ============================================================
        // CPU 路径 - 使用 Pinned Memory 加速 H2D 传输
        // ============================================================
        if (cpu_batch > 0) {
            LOG_DEBUG_FMT("[DetectionInfer] CPU path: {} frames (pinned memory)", cpu_batch);

            int hw = input_height_ * input_width_;
            int batch_stride = 3 * hw;
            size_t alloc_size = cpu_batch * batch_stride * sizeof(float);

            // RAII 管理临时缓冲区，避免异常时泄露
            std::unique_ptr<float, decltype(&free)> host_input_heap(
                h_pinned_input_ ? nullptr : static_cast<float*>(malloc(alloc_size)),
                free
            );

            float* host_input = h_pinned_input_ ? h_pinned_input_ : host_input_heap.get();

            for (int i = 0; i < cpu_batch; ++i) {
                int orig_idx = cpu_indices[i];
                const cv::Mat& image = *frames[orig_idx]->mat;
                const float* img_ptr = image.ptr<float>();
                float* batch_ptr = host_input + i * batch_stride;

                // HWC → NCHW
                for (int h = 0; h < input_height_; ++h) {
                    for (int w = 0; w < input_width_; ++w) {
                        int src_idx = (h * input_width_ + w) * 3;
                        int dst_idx = h * input_width_ + w;
                        batch_ptr[0 * hw + dst_idx] = img_ptr[src_idx + 2]; // R
                        batch_ptr[1 * hw + dst_idx] = img_ptr[src_idx + 1]; // G
                        batch_ptr[2 * hw + dst_idx] = img_ptr[src_idx + 0]; // B
                    }
                }
            }

            size_t gpu_offset = gpu_batch * batch_stride * sizeof(float);
            cudaMemcpyAsync(static_cast<char*>(d_input_) + gpu_offset,
                            host_input,
                            alloc_size,
                            cudaMemcpyHostToDevice, transfer_stream_);
        }

        // 等待传输完成后再推理
        cudaStreamSynchronize(transfer_stream_);

        // 设置 Tensor 地址并执行推理（通过 HAL 接口）
        engine_->setInputTensor(input_name_, d_input_);
        engine_->setOutputTensor(boxes_name_, d_boxes_);
        engine_->setOutputTensor(scores_name_, d_scores_);
        engine_->setOutputTensor(classes_name_, d_classes_);
        engine_->setOutputTensor(batch_ids_name_, d_batch_ids_);
        engine_->setOutputTensor(num_dets_name_, d_num_dets_);

        // ============================================================
        // 推理执行 - 优先使用 CUDA Graph
        // ============================================================
        auto t0 = std::chrono::high_resolution_clock::now();

        if (cuda_graph_ready_ && cuda_graph_batch_size_ == valid_batch) {
            if (!executeCudaGraph()) {
                LOG_WARN_FMT("[DetectionInfer] CUDA Graph execution failed, fallback to normal");
                if (!engine_->inferAsync(compute_stream_)) {
                    LOG_ERROR_FMT("[DetectionInfer] inferAsync failed for batch={}", valid_batch);
                    return results;
                }
                engine_->synchronize(compute_stream_);
            }
        } else {
            if (!engine_->inferAsync(compute_stream_)) {
                LOG_ERROR_FMT("[DetectionInfer] inferAsync failed for batch={}", valid_batch);
                return results;
            }
            engine_->synchronize(compute_stream_);
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        LOG_DEBUG_FMT("[DetectionInfer] Inference time: {:.2f}ms (batch={})", infer_ms, valid_batch);

        // ============================================================
        // 后处理 - 使用 Pinned Memory 加速 D2H 传输
        // ============================================================
        int64_t* num_dets_ptr = h_pinned_num_dets_ ? h_pinned_num_dets_ : &h_num_dets_;
        float* boxes_ptr = h_pinned_boxes_ ? h_pinned_boxes_ : h_boxes_.data();
        float* scores_ptr = h_pinned_scores_ ? h_pinned_scores_ : h_scores_.data();
        int64_t* classes_ptr = h_pinned_classes_ ? h_pinned_classes_ : h_classes_.data();
        int64_t* batch_ids_ptr = h_pinned_batch_ids_ ? h_pinned_batch_ids_ : h_batch_ids_.data();

        // 异步 D2H 传输
        cudaMemcpyAsync(num_dets_ptr, d_num_dets_, sizeof(int64_t),
                        cudaMemcpyDeviceToHost, transfer_stream_);
        cudaStreamSynchronize(transfer_stream_);

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

        cudaMemcpyAsync(boxes_ptr, d_boxes_, actual_boxes, cudaMemcpyDeviceToHost, transfer_stream_);
        cudaMemcpyAsync(scores_ptr, d_scores_, actual_scores, cudaMemcpyDeviceToHost, transfer_stream_);
        cudaMemcpyAsync(classes_ptr, d_classes_, actual_classes, cudaMemcpyDeviceToHost, transfer_stream_);
        cudaMemcpyAsync(batch_ids_ptr, d_batch_ids_, actual_batch_ids, cudaMemcpyDeviceToHost, transfer_stream_);
        cudaStreamSynchronize(transfer_stream_);

        // 直接把 D2H 结果指针传给 postprocessBatch，省去 pinned -> h_* 向量的二次拷贝

        // 构建 scale 数组
        std::vector<float> valid_scale_x(valid_batch);
        std::vector<float> valid_scale_y(valid_batch);
        std::vector<float> valid_letter_scale(valid_batch, 1.0f);
        std::vector<int> valid_letter_pad_x(valid_batch, 0);
        std::vector<int> valid_letter_pad_y(valid_batch, 0);
        std::vector<int> valid_letterbox_used(valid_batch, 0);

        int idx = 0;
        for (int i : gpu_indices) {
            valid_scale_x[idx] = scale_x[i];
            valid_scale_y[idx] = scale_y[i];
            valid_letter_scale[idx] = frames[i]->letter_scale;
            valid_letter_pad_x[idx] = frames[i]->letter_pad_x;
            valid_letter_pad_y[idx] = frames[i]->letter_pad_y;
            valid_letterbox_used[idx] = frames[i]->letterbox_used ? 1 : 0;
            idx++;
        }
        for (int i : cpu_indices) {
            valid_scale_x[idx] = scale_x[i];
            valid_scale_y[idx] = scale_y[i];
            valid_letter_scale[idx] = frames[i]->letter_scale;
            valid_letter_pad_x[idx] = frames[i]->letter_pad_x;
            valid_letter_pad_y[idx] = frames[i]->letter_pad_y;
            valid_letterbox_used[idx] = frames[i]->letterbox_used ? 1 : 0;
            idx++;
        }

        auto all_detections = postprocessBatch(valid_batch, total_dets,
                                                valid_scale_x.data(), valid_scale_y.data(), confidence_threshold_,
                                                valid_letter_scale.data(),
                                                valid_letter_pad_x.data(),
                                                valid_letter_pad_y.data(),
                                                valid_letterbox_used.data(),
                                                boxes_ptr, scores_ptr, classes_ptr, batch_ids_ptr);

        // 映射回原始帧索引
        idx = 0;
        for (int i : gpu_indices) {
            results[i]->detections = std::move(all_detections[idx++]);
        }
        for (int i : cpu_indices) {
            results[i]->detections = std::move(all_detections[idx++]);
        }

        LOG_INFO_FMT("[DetectionInfer] Batch done: gpu={} cpu={} valid_batch={} total_dets={}, detections={}",
                     gpu_batch, cpu_batch, valid_batch, total_dets, results[0]->detections.size());
#else
        // ============================================================
        // 主机引擎路径（RKNN / Ascend 等）：逐帧推理，引擎自管输出
        // ============================================================
        const int hw = input_height_ * input_width_;
        const size_t frame_floats = static_cast<size_t>(3) * hw;
        h_input_host_.resize(static_cast<size_t>(valid_batch) * frame_floats);

        std::vector<float> all_boxes, all_scores;
        std::vector<int64_t> all_classes, all_batch_ids;
        std::vector<float> v_scale_x(valid_batch), v_scale_y(valid_batch);
        std::vector<float> v_ls(valid_batch, 1.0f);
        std::vector<int> v_px(valid_batch, 0), v_py(valid_batch, 0), v_lu(valid_batch, 0);

        int slot = 0;
        for (int b = 0; b < actual_batch && slot < valid_batch; ++b) {
            // ---- NV12 直通分支（NV12 输入模型）----
            if (input_nv12_) {
                auto& fr = frames[b];
                if (fr && fr->is_nv12 && fr->nv12 &&
                    fr->nv12_width > 0 && fr->nv12_height > 0) {
                    if (nv12_engine_ && nv12_engine_->setNv12Input(fr->nv12->data(), fr->nv12_width, fr->nv12_height) &&
                        engine_->infer()) {
                        const int sw = fr->nv12_width, sh = fr->nv12_height;
                        const float sc = std::min(static_cast<float>(input_width_) / sw,
                                                  static_cast<float>(input_height_) / sh);
                        const int lw = std::max(2, static_cast<int>(std::round(sw * sc))) & ~1;
                        const int lh = std::max(2, static_cast<int>(std::round(sh * sc))) & ~1;
                        fr->width = sw;
                        fr->height = sh;
                        fr->letterbox_used = true;
                        fr->letter_scale = sc;
                        fr->letter_pad_x = ((input_width_ - lw) / 2) & ~1;
                        fr->letter_pad_y = ((input_height_ - lh) / 2) & ~1;

                        const int64_t n = *static_cast<const int64_t*>(engine_->getOutputTensor(num_dets_name_));
                        const int det_n = static_cast<int>(std::min<int64_t>(n, MAX_DETS));
                        const float* nbl = static_cast<const float*>(engine_->getOutputTensor(boxes_name_));
                        const float* nbs = static_cast<const float*>(engine_->getOutputTensor(scores_name_));
                        const int64_t* nbc = static_cast<const int64_t*>(engine_->getOutputTensor(classes_name_));
                        for (int i = 0; i < det_n; ++i) {
                            all_boxes.insert(all_boxes.end(), nbl + i * 4, nbl + i * 4 + 4);
                            all_scores.push_back(nbs[i]);
                            all_classes.push_back(nbc[i]);
                            all_batch_ids.push_back(slot);
                        }
                        v_scale_x[slot] = 1.0f;
                        v_scale_y[slot] = 1.0f;
                        v_ls[slot] = fr->letter_scale;
                        v_px[slot] = fr->letter_pad_x;
                        v_py[slot] = fr->letter_pad_y;
                        v_lu[slot] = 1;
                        ++slot;
                        continue;
                    }
                    LOG_WARN_FMT("[DetectionInfer] NV12 infer failed for frame {}", b);
                    continue;
                }
            }
            if (!(frames[b]->mat && !frames[b]->mat->empty() &&
                  frames[b]->mat->type() == CV_32FC3)) {
                continue;
            }
            const cv::Mat& m = *frames[b]->mat;
            if (m.cols != input_width_ || m.rows != input_height_) {
                LOG_WARN_FMT("[DetectionInfer] Frame[{}] size {}x{} != model input {}x{}, skip",
                             b, m.cols, m.rows, input_width_, input_height_);
                continue;
            }

            // HWC CV_32FC3 (BGR) → NCHW RGB
            float* dst = h_input_host_.data() + static_cast<size_t>(slot) * frame_floats;
            const float* img = m.ptr<float>();
            for (int y = 0; y < input_height_; ++y) {
                for (int x = 0; x < input_width_; ++x) {
                    const int src_idx = (y * input_width_ + x) * 3;
                    const int dst_idx = y * input_width_ + x;
                    dst[0 * hw + dst_idx] = img[src_idx + 2];  // R
                    dst[1 * hw + dst_idx] = img[src_idx + 1];  // G
                    dst[2 * hw + dst_idx] = img[src_idx + 0];  // B
                }
            }

            engine_->setInputTensor(input_name_, dst);
            if (!engine_->infer()) {
                LOG_ERROR_FMT("[DetectionInfer] Host engine infer failed for frame {}", b);
                return results;
            }

            const int64_t n = *static_cast<const int64_t*>(engine_->getOutputTensor(num_dets_name_));
            const int det_n = static_cast<int>(std::min<int64_t>(n, MAX_DETS));
            const float* boxes = static_cast<const float*>(engine_->getOutputTensor(boxes_name_));
            const float* scores = static_cast<const float*>(engine_->getOutputTensor(scores_name_));
            const int64_t* classes = static_cast<const int64_t*>(engine_->getOutputTensor(classes_name_));

            for (int i = 0; i < det_n; ++i) {
                all_boxes.insert(all_boxes.end(), boxes + i * 4, boxes + i * 4 + 4);
                all_scores.push_back(scores[i]);
                all_classes.push_back(classes[i]);
                all_batch_ids.push_back(slot);
            }

            // 记录该槽位的坐标反变换参数
            v_scale_x[slot] = scale_x[b];
            v_scale_y[slot] = scale_y[b];
            v_ls[slot] = frames[b]->letter_scale;
            v_px[slot] = frames[b]->letter_pad_x;
            v_py[slot] = frames[b]->letter_pad_y;
            v_lu[slot] = frames[b]->letterbox_used ? 1 : 0;
            ++slot;
        }

        if (slot == 0) {
            LOG_WARN_FMT("[DetectionInfer] No valid frames in batch (host path)");
            return results;
        }

        // 复用成员 h_* 缓冲供 postprocessBatch 消费
        const int total_dets = static_cast<int>(all_scores.size());
        h_boxes_.assign(all_boxes.begin(), all_boxes.end());
        h_scores_.assign(all_scores.begin(), all_scores.end());
        h_classes_.assign(all_classes.begin(), all_classes.end());
        h_batch_ids_.assign(all_batch_ids.begin(), all_batch_ids.end());
        h_num_dets_ = total_dets;

        auto all_detections = postprocessBatch(slot, total_dets,
                                               v_scale_x.data(), v_scale_y.data(),
                                               confidence_threshold_,
                                               v_ls.data(), v_px.data(), v_py.data(),
                                               v_lu.data());

        int out_idx = 0;
        for (int b = 0; b < actual_batch; ++b) {
            if (frames[b]->mat && !frames[b]->mat->empty() &&
                frames[b]->mat->type() == CV_32FC3 && frames[b]->mat->cols == input_width_) {
                results[b]->detections = std::move(all_detections[out_idx++]);
            }
        }

        LOG_INFO_FMT("[DetectionInfer] Host batch done: frames={} total_dets={}",
                     slot, total_dets);
        return results;
#endif // 主机引擎路径结束

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] Batch inference exception: {}", e.what());
    }

    return results;
}

// ============================================================
// ============================================================
// 后处理 - 支持 letterbox 反变换
// ============================================================
std::vector<std::vector<core::InferenceResultPacket::BBox>> DetectionInferNode::postprocessBatch(
    int batch_size, int total_dets,
    const float scale_x[], const float scale_y[], float conf_thresh,
    const float letter_scale[],
    const int letter_pad_x[],
    const int letter_pad_y[],
    const int letterbox_used[],
    const float* boxes,
    const float* scores,
    const int64_t* classes,
    const int64_t* batch_ids) {

    // 优先使用调用方直接给出的 D2H 结果（如 pinned 缓冲），避免经成员向量二次拷贝
    const float* boxes_p = boxes ? boxes : h_boxes_.data();
    const float* scores_p = scores ? scores : h_scores_.data();
    const int64_t* classes_p = classes ? classes : h_classes_.data();
    const int64_t* batch_ids_p = batch_ids ? batch_ids : h_batch_ids_.data();

    std::vector<std::vector<core::InferenceResultPacket::BBox>> all_detections(batch_size);

    for (int i = 0; i < total_dets; ++i) {
        float score = scores_p[i];
        if (score < conf_thresh) continue;

        int batch_id = static_cast<int>(batch_ids_p[i]);
        if (batch_id < 0 || batch_id >= batch_size) {
            LOG_WARN_FMT("[DetectionInfer] Invalid batch_id {} at det {}, max={}", batch_id, i, batch_size);
            continue;
        }

        float cx = boxes_p[i * 4 + 0];
        float cy = boxes_p[i * 4 + 1];
        float w  = boxes_p[i * 4 + 2];
        float h  = boxes_p[i * 4 + 3];

        core::InferenceResultPacket::BBox box;

        // Letterbox 反变换
        if (letterbox_used && letterbox_used[batch_id] && letter_scale) {
            float inv_scale = 1.0f / letter_scale[batch_id];
            float pad_x = static_cast<float>(letter_pad_x[batch_id]);
            float pad_y = static_cast<float>(letter_pad_y[batch_id]);

            float orig_cx = (cx - pad_x) * inv_scale;
            float orig_cy = (cy - pad_y) * inv_scale;
            float orig_w  = w * inv_scale;
            float orig_h  = h * inv_scale;

            box.x = static_cast<int>(orig_cx - orig_w / 2.0f);
            box.y = static_cast<int>(orig_cy - orig_h / 2.0f);
            box.w = static_cast<int>(orig_w);
            box.h = static_cast<int>(orig_h);
        } else {
            box.x = static_cast<int>((cx - w / 2.0f) * scale_x[batch_id]);
            box.y = static_cast<int>((cy - h / 2.0f) * scale_y[batch_id]);
            box.w = static_cast<int>(w * scale_x[batch_id]);
            box.h = static_cast<int>(h * scale_y[batch_id]);
        }

        box.confidence = score;
        box.class_id = static_cast<int>(classes_p[i]);

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
// CUDA Graph 捕获 - 消除 kernel launch overhead
// ============================================================
#ifdef WITH_CUDA
bool DetectionInferNode::captureCudaGraph(int batch_size) {
#ifdef WITH_TENSORRT
    auto* raw_context = graph_engine_ ? static_cast<nvinfer1::IExecutionContext*>(graph_engine_->getRawContext()) : nullptr;
    if (!raw_context) return false;

    destroyCudaGraph();

    try {
        nvinfer1::Dims4 input_dims(batch_size, 3, input_height_, input_width_);
        if (!raw_context->setInputShape(input_name_.c_str(), input_dims)) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph: setInputShape failed");
            return false;
        }

        // 设置 tensor 地址
        engine_->setInputTensor(input_name_, d_input_);
        engine_->setOutputTensor(boxes_name_, d_boxes_);
        engine_->setOutputTensor(scores_name_, d_scores_);
        engine_->setOutputTensor(classes_name_, d_classes_);
        engine_->setOutputTensor(batch_ids_name_, d_batch_ids_);
        engine_->setOutputTensor(num_dets_name_, d_num_dets_);

        // TRT 10.3: 使用 updateDeviceMemorySizeForShapes 获取实际所需大小（含 activation + scratch）
        size_t actual_mem_size = graph_engine_->updateDeviceMemorySizeForShapes();
        size_t static_mem_size = graph_engine_->getDeviceMemorySize();
        size_t workspace_size = (actual_mem_size > 0) ? actual_mem_size : static_mem_size;

        LOG_INFO_FMT("[DetectionInfer] CUDA Graph: workspace size: static={}KB, actual={}KB",
                     static_mem_size / 1024, actual_mem_size / 1024);

        // 预分配 workspace，避免 enqueueV3 在 capture 期间调用 cudaMallocAsync
        void* d_workspace = nullptr;
        if (workspace_size > 0) {
            // 多分配 256MB 作为安全余量，应对 TRT 内部额外分配
            size_t alloc_size = workspace_size + 256 * 1024 * 1024;
            cudaError_t err = cudaMalloc(&d_workspace, alloc_size);
            if (err != cudaSuccess) {
                LOG_ERROR_FMT("[DetectionInfer] CUDA Graph: cudaMalloc workspace failed ({}KB): {}",
                              alloc_size / 1024, cudaGetErrorString(err));
                return false;
            }
            // 使用 setDeviceMemoryV2 设置内存和大小
            if (!graph_engine_->setDeviceMemoryV2(d_workspace, static_cast<int64_t>(alloc_size))) {
                LOG_ERROR_FMT("[DetectionInfer] CUDA Graph: setDeviceMemoryV2 failed");
                cudaFree(d_workspace);
                return false;
            }
        }

        // warmup: enqueueV3 首次调用可能触发内部初始化
        if (!raw_context->enqueueV3(compute_stream_)) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph warmup enqueueV3 failed");
            if (d_workspace) cudaFree(d_workspace);
            return false;
        }
        cudaStreamSynchronize(compute_stream_);

        // 二次 warmup: 确保所有内部状态已初始化
        if (!raw_context->enqueueV3(compute_stream_)) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph second warmup enqueueV3 failed");
            if (d_workspace) cudaFree(d_workspace);
            return false;
        }
        cudaStreamSynchronize(compute_stream_);

        // 开始捕获 CUDA Graph
        cudaError_t begin_err = cudaStreamBeginCapture(compute_stream_, cudaStreamCaptureModeGlobal);
        if (begin_err != cudaSuccess) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph: beginCapture failed: {} — disabling CUDA Graph",
                          cudaGetErrorString(begin_err));
            if (d_workspace) cudaFree(d_workspace);
            return false;
        }

        // 执行推理（会被捕获到 graph 中）
        if (!raw_context->enqueueV3(compute_stream_)) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph capture: enqueueV3 failed");
            // 必须先结束捕获使流恢复，再销毁已捕获的 graph
            cudaGraph_t partial = nullptr;
            cudaStreamEndCapture(compute_stream_, &partial);
            if (partial) cudaGraphDestroy(partial);
            if (d_workspace) cudaFree(d_workspace);
            return false;
        }

        // 结束捕获
        cudaError_t err = cudaStreamEndCapture(compute_stream_, &cuda_graph_);
        if (err != cudaSuccess || !cuda_graph_) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph capture failed: {}", cudaGetErrorString(err));
            if (cuda_graph_) {
                cudaGraphDestroy(cuda_graph_);
                cuda_graph_ = nullptr;
            }
            if (d_workspace) cudaFree(d_workspace);
            return false;
        }

        // 实例化 graph
        err = cudaGraphInstantiate(&cuda_graph_exec_, cuda_graph_, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            LOG_ERROR_FMT("[DetectionInfer] CUDA Graph instantiate failed: {}", cudaGetErrorString(err));
            if (cuda_graph_) {
                cudaGraphDestroy(cuda_graph_);
                cuda_graph_ = nullptr;
            }
            if (d_workspace) cudaFree(d_workspace);
            return false;
        }

        // workspace 指针需要在 graph 执行期间保持有效，不能在这里释放
        // 它会在 destroyCudaGraph() 时释放
        cuda_graph_workspace_ = d_workspace;
        cuda_graph_batch_size_ = batch_size;
        cuda_graph_ready_ = true;

        LOG_INFO_FMT("[DetectionInfer] CUDA Graph captured and instantiated for batch_size={} (workspace={}KB, alloc={}KB)",
                     batch_size, workspace_size / 1024, (workspace_size + 256 * 1024 * 1024) / 1024);
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] CUDA Graph capture exception: {}", e.what());
        return false;
    }
#else
    (void)batch_size;
    return false;
#endif
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
    if (cuda_graph_workspace_) {
        cudaFree(cuda_graph_workspace_);
        cuda_graph_workspace_ = nullptr;
    }
    cuda_graph_ready_ = false;
    cuda_graph_batch_size_ = 0;
}

// ============================================================
// Pinned Memory 管理
// ============================================================
bool DetectionInferNode::allocatePinnedMemory() {
    int max_batch = max_batch_size_.load();
    int hw = input_height_ * input_width_;
    int batch_stride = 3 * hw;

    size_t input_bytes = static_cast<size_t>(max_batch) * batch_stride * sizeof(float);
    size_t boxes_bytes = static_cast<size_t>(max_batch) * MAX_DETS * 4 * sizeof(float);
    size_t scores_bytes = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(float);
    size_t classes_bytes = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);
    size_t batch_ids_bytes = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);

    cudaError_t err;

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
#endif // WITH_CUDA

// ============================================================
// 初始化引擎（通过 HAL 工厂创建；CUDA/主机路径见函数内条件编译）
// ============================================================
bool DetectionInferNode::initEngine(const std::string& engine_path) {
    try {
        // 通过 HAL 工厂创建推理引擎
        engine_ = hal::DetectionInferenceEngineFactory::instance().create(backend_type_);
        if (!engine_) {
            // 平台无推理后端（如纯 CPU / RKNN 构建未启用）：以 mock 模式
            // 运行（processBatch 生成假检测框），保证流水线可构建可验证
            LOG_WARN("[DetectionInfer] No inference backend available on this "
                     "platform, running in mock mode");
            return true;
        }
        // 探测可选能力接口（CUDA Graph/NV12 直通），不支持则为 nullptr
        graph_engine_ = dynamic_cast<hal::IGraphCapturable*>(engine_.get());
        nv12_engine_ = dynamic_cast<hal::INv12Input*>(engine_.get());

        hal::DetectionInferenceConfig config;
        config.model_path = engine_path;
        config.input_width = input_width_;
        config.input_height = input_height_;
        config.max_batch_size = max_batch_size_.load();
        config.max_detections = MAX_DETS;
        config.precision = precision_;
        config.device_id = device_id_;
        config.enable_cuda_graph = cuda_graph_enabled_.load();

        if (!engine_->loadModel(config)) {
            LOG_ERROR_FMT("[DetectionInfer] Failed to load model: {}", engine_path);
            engine_.reset();
            return false;
        }

        LOG_INFO_FMT("[DetectionInfer] Engine loaded via backend: {}", engine_->getBackendName());

        int max_batch = max_batch_size_.load();
        out_boxes_size_ = static_cast<size_t>(max_batch) * MAX_DETS * 4 * sizeof(float);
        out_scores_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(float);
        out_classes_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);
        out_batch_ids_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);
        out_num_dets_size_ = sizeof(int64_t);

#ifdef WITH_CUDA
        // 创建双流架构
        cudaStreamCreateWithFlags(&compute_stream_, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&transfer_stream_, cudaStreamNonBlocking);

        // 计算缓冲区大小
        input_size_ = static_cast<size_t>(max_batch) * 3 * input_height_ * input_width_ * sizeof(float);
        // 分配 GPU 缓冲区
        if (!cudaMallocChecked(&d_input_, input_size_, "d_input_") ||
            !cudaMallocChecked(&d_boxes_, out_boxes_size_, "d_boxes_") ||
            !cudaMallocChecked(&d_scores_, out_scores_size_, "d_scores_") ||
            !cudaMallocChecked(&d_classes_, out_classes_size_, "d_classes_") ||
            !cudaMallocChecked(&d_batch_ids_, out_batch_ids_size_, "d_batch_ids_") ||
            !cudaMallocChecked(&d_num_dets_, out_num_dets_size_, "d_num_dets_")) {
            LOG_ERROR("[DetectionInfer] Failed to allocate GPU buffers");
            return false;
        }

        // 分配 Pinned Memory
        allocatePinnedMemory();

        // 预捕获 CUDA Graph
        if (cuda_graph_enabled_) {
            captureCudaGraph(max_batch);
        }

        LOG_INFO_FMT("[DetectionInfer] Engine initialized: {} (max_batch={}, max_dets={}, streams=2, pinned={}, cuda_graph={})",
                     engine_path, max_batch, MAX_DETS,
                     h_pinned_input_ ? "ON" : "OFF",
                     cuda_graph_ready_ ? "READY" : "OFF");
#else
        // 主机引擎路径（RKNN / Ascend 等）：引擎自管输出内存，
        // 节点侧仅准备 host NCHW 输入缓冲
        h_input_host_.resize(static_cast<size_t>(max_batch) * 3 * input_height_ * input_width_);
        engine_->allocateOutputBuffers();
        LOG_INFO_FMT("[DetectionInfer] Engine initialized (host path): {} (max_batch={}, max_dets={})",
                     engine_path, max_batch, MAX_DETS);
#endif

        // 分配 CPU fallback 缓冲区（两种路径共用）
        h_boxes_.resize(max_batch * MAX_DETS * 4);
        h_scores_.resize(max_batch * MAX_DETS);
        h_classes_.resize(max_batch * MAX_DETS);
        h_batch_ids_.resize(max_batch * MAX_DETS);
        h_num_dets_ = 0;

        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] initEngine exception: {}", e.what());
        return false;
    }
}

#ifdef WITH_CUDA
bool DetectionInferNode::cudaMallocChecked(void** ptr, size_t size, const char* name) {
    cudaError_t err = cudaMalloc(ptr, size);
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("[DetectionInfer] cudaMalloc failed for {}: {}", name, cudaGetErrorString(err));
        return false;
    }
    return true;
}
#endif // WITH_CUDA


REGISTER_NODE("detection_infer", DetectionInferNode)

} // namespace nodes
} // namespace ai_stream
