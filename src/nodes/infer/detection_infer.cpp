// src/nodes/infer/detection_infer.cpp
#include "detection_infer.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "tensor_rt_logger.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

#include <NvInfer.h>

namespace ai_stream {
namespace nodes {

// ============================================================
// DetectionInferNode
// ============================================================

DetectionInferNode::DetectionInferNode() : IInferNode("DetectionInfer") {
    LOG_DEBUG_FMT("[DetectionInfer] Constructor");
}

DetectionInferNode::~DetectionInferNode() {
    stop();
    
    if (d_input_) cudaFree(d_input_);
    if (d_boxes_) cudaFree(d_boxes_);
    if (d_scores_) cudaFree(d_scores_);
    if (d_classes_) cudaFree(d_classes_);
    if (d_batch_ids_) cudaFree(d_batch_ids_);
    if (d_num_dets_) cudaFree(d_num_dets_);
    if (d_preprocess_tmp_) cudaFree(d_preprocess_tmp_);
    if (stream_) cudaStreamDestroy(stream_);
    
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
    LOG_INFO_FMT("[DetectionInfer] Started with max_batch={}", max_batch_size_.load());
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
// 推理主循环
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
// 多 batch 推理核心（双路径适配）
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
        std::vector<int> gpu_indices;      // 使用 frame->d_ptr 的帧索引
        std::vector<int> cpu_indices;      // 使用 frame->mat 的帧索引
        std::vector<void*> d_ptrs;         // GPU 指针列表
        std::vector<size_t> d_pitches;     // GPU pitch 列表

        float scale_x[actual_batch];
        float scale_y[actual_batch];

        for (int b = 0; b < actual_batch; ++b) {
            // 计算缩放比例
            if (frames[b]->source_mat && !frames[b]->source_mat->empty()) {
                scale_x[b] = static_cast<float>(frames[b]->source_mat->cols) / INPUT_W;
                scale_y[b] = static_cast<float>(frames[b]->source_mat->rows) / INPUT_H;
            LOG_INFO_FMT("[TensorRT] source_mat size: {}x{}", frames[b]->source_mat->cols, frames[b]->source_mat->rows);
            } else if (frames[b]->mat && !frames[b]->mat->empty()) {
                scale_x[b] = static_cast<float>(frames[b]->mat->cols) / INPUT_W;
                scale_y[b] = static_cast<float>(frames[b]->mat->rows) / INPUT_H;
                LOG_INFO_FMT("[TensorRT] mat size: {}x{}", frames[b]->mat->cols, frames[b]->mat->rows);
            } else {
                scale_x[b] = frames[b]->width / static_cast<float>(INPUT_W);
                scale_y[b] = frames[b]->height / static_cast<float>(INPUT_H);
                LOG_INFO_FMT("[TensorRT] frame size: {}x{}", frames[b]->width, frames[b]->height);
            }

            // 【关键】判断数据来源
            if (frames[b]->is_gpu && frames[b]->d_ptr) {
                // 硬件解码/GPU 预处理路径：数据已在 GPU
                gpu_indices.push_back(b);
                d_ptrs.push_back(frames[b]->d_ptr);
                d_pitches.push_back(frames[b]->d_pitch);
            } else {
                // 软件路径：数据在 CPU cv::Mat
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
        // 【GPU 路径】直接复用上游 GPU 指针
        // ============================================================
        if (gpu_batch > 0) {
            LOG_DEBUG_FMT("[DetectionInfer] GPU path: {} frames", gpu_batch);
            
            // 检查是否需要连续内存拷贝（TensorRT 通常需要连续 NCHW）
            // 如果上游已经是 NCHW 且连续，可以直接用；否则需要 cudaMemcpy2DAsync 整理
            bool all_continuous = true;
            for (int i = 0; i < gpu_batch; ++i) {
                if (d_pitches[i] != INPUT_W * 3 * sizeof(float)) {
                    all_continuous = false;
                    break;
                }
            }

            if (all_continuous && gpu_batch == valid_batch) {
                // 最优路径：所有帧都是连续 NCHW，直接拼接指针
                // 注意：TensorRT 需要 batch 维度的连续内存
                // 如果上游是分开分配的，仍需拷贝到 d_input_
                for (int i = 0; i < gpu_batch; ++i) {
                    size_t offset = i * 3 * INPUT_H * INPUT_W * sizeof(float);
                    cudaMemcpyAsync(static_cast<char*>(d_input_) + offset,
                                    d_ptrs[i],
                                    3 * INPUT_H * INPUT_W * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream_);
                }
            } else {
                // 需要整理内存布局
                for (int i = 0; i < gpu_batch; ++i) {
                    size_t offset = i * 3 * INPUT_H * INPUT_W * sizeof(float);
                    cudaMemcpyAsync(static_cast<char*>(d_input_) + offset,
                                    d_ptrs[i],
                                    3 * INPUT_H * INPUT_W * sizeof(float),
                                    cudaMemcpyDeviceToDevice, stream_);
                }
            }
        }

        // ============================================================
        // 【CPU 路径】H2D 上传
        // ============================================================
        if (cpu_batch > 0) {
            LOG_DEBUG_FMT("[DetectionInfer] CPU path: {} frames", cpu_batch);
            
            int hw = INPUT_H * INPUT_W;
            int batch_stride = 3 * hw;
            
            // 预分配 host 缓冲区
            static thread_local std::vector<float> host_input;
            host_input.resize(cpu_batch * batch_stride);

            for (int i = 0; i < cpu_batch; ++i) {
                int orig_idx = cpu_indices[i];
                const cv::Mat& image = *frames[orig_idx]->mat;
                
                // CV_32FC3 已经是 float，直接按通道拆分
                // OpenCV 的 CV_32FC3 是 HWC 格式，需要转 NCHW
                const float* img_ptr = image.ptr<float>();
                float* batch_ptr = host_input.data() + i * batch_stride;

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
            cudaMemcpyAsync(static_cast<char*>(d_input_) + gpu_offset,
                            host_input.data(),
                            cpu_batch * batch_stride * sizeof(float),
                            cudaMemcpyHostToDevice, stream_);
        }

        cudaStreamSynchronize(stream_);

        // 设置 Tensor 地址并执行推理
        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(boxes_name_.c_str(), d_boxes_);
        context_->setTensorAddress(scores_name_.c_str(), d_scores_);
        context_->setTensorAddress(classes_name_.c_str(), d_classes_);
        context_->setTensorAddress(batch_ids_name_.c_str(), d_batch_ids_);
        context_->setTensorAddress(num_dets_name_.c_str(), d_num_dets_);

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!context_->enqueueV3(stream_)) {
            LOG_ERROR_FMT("[DetectionInfer] enqueueV3 failed for batch={}", valid_batch);
            return results;
        }
        cudaStreamSynchronize(stream_);
        auto t1 = std::chrono::high_resolution_clock::now();

        // 后处理（与之前相同）
        cudaMemcpyAsync(&h_num_dets_, d_num_dets_, sizeof(int64_t),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        int total_dets = static_cast<int>(h_num_dets_);
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

        cudaMemcpyAsync(h_boxes_.data(), d_boxes_, actual_boxes, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_scores_.data(), d_scores_, actual_scores, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_classes_.data(), d_classes_, actual_classes, cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_batch_ids_.data(), d_batch_ids_, actual_batch_ids, cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 构建 scale 数组（注意：batch_id 对应 valid_batch 中的顺序）
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

        LOG_INFO_FMT("[DetectionInfer] Batch done: gpu={} cpu={} total_valid={} total={},detections={}",
                     gpu_batch, cpu_batch, valid_batch,total_dets,results[0]->detections.size());

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] Batch inference exception: {}", e.what());
    }

    return results;
}

// ============================================================
// 【保留】CPU 路径预处理（软件解码 fallback）
// ============================================================
void DetectionInferNode::preprocessBatchCpu(const std::vector<cv::Mat*>& images,
                                             float* gpu_buffer, int batch_size) {
    int hw = INPUT_H * INPUT_W;
    int batch_stride = 3 * hw;
    
    static thread_local std::vector<float> host_input;
    host_input.resize(batch_size * batch_stride);

    for (int b = 0; b < batch_size; ++b) {
        const cv::Mat& image = *images[b];
        const float* img_ptr = image.ptr<float>();
        float* batch_ptr = host_input.data() + b * batch_stride;

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

    cudaMemcpyAsync(gpu_buffer, host_input.data(),
                    batch_size * batch_stride * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
}

// ============================================================
// 后处理（不变）
// ============================================================
std::vector<std::vector<core::InferenceResultPacket::BBox>> DetectionInferNode::postprocessBatch(
    int batch_size, int total_dets, const float scale_x[], const float scale_y[], float conf_thresh) {
    
    std::vector<std::vector<core::InferenceResultPacket::BBox>> all_detections(batch_size);

    for (int i = 0; i < total_dets; ++i) {
        float score = h_scores_[i];
        if (score < conf_thresh) continue;

        // 从 det_batch_ids 获取该检测属于哪个 batch
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
// 初始化引擎（补充 d_preprocess_tmp_ 分配）
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

        cudaStreamCreate(&stream_);

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

        // 【新增】预分配 GPU 预处理临时缓冲区
        cudaMalloc(&d_preprocess_tmp_, input_size_);
        d_preprocess_tmp_size_ = input_size_;

        h_boxes_.resize(max_batch * MAX_DETS * 4);
        h_scores_.resize(max_batch * MAX_DETS);
        h_classes_.resize(max_batch * MAX_DETS);
        h_batch_ids_.resize(max_batch * MAX_DETS);
        h_num_dets_ = 0;

        LOG_INFO_FMT("[DetectionInfer] Engine loaded: {} (max_batch={}, max_dets={})",
                     engine_path, max_batch, MAX_DETS);
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] initEngine exception: {}", e.what());
        return false;
    }
}

REGISTER_NODE("detection_infer", DetectionInferNode)

} // namespace nodes
} // namespace ai_stream