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
    if (d_batch_ids_) cudaFree(d_batch_ids_);  // 新增
    if (d_num_dets_) cudaFree(d_num_dets_);
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
    {
        std::lock_guard<std::mutex> lock(batch_mutex_);
        batch_cv_.notify_all();
    }
    if (worker_.joinable()) {
        worker_.join();
    }
    LOG_INFO_FMT("[DetectionInfer] Stopped");
}

void DetectionInferNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
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

        // 1. 等待第一帧（阻塞等待）
        std::shared_ptr<core::VideoFramePacket> first_frame;
        if (!queue_.pop(first_frame, std::chrono::milliseconds(batch_timeout_ms_))) {
            continue;
        }
        batch_frames.push_back(first_frame);

        auto batch_start_time = std::chrono::high_resolution_clock::now();

        // 2. 尝试收集更多帧凑齐 batch（带超时）
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
        
        // 3. 执行推理
        auto t0 = std::chrono::high_resolution_clock::now();
        auto results = processBatch(batch_frames);
        auto t1 = std::chrono::high_resolution_clock::now();
        
        float batch_infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        LOG_INFO_FMT("[DetectionInfer] Batch inference: {} frames, total={:.2f}ms, avg={:.2f}ms/frame",
                     actual_batch, batch_infer_ms, batch_infer_ms / actual_batch);

        // 4. 广播结果
        for (auto& result : results) {
            if (result) {
                result->cost_ms = utils::TimeUtil::currentTimeMs() - in_time_ms_;
                result->cost_time_map = first_frame->cost_time_map;
                result->cost_time_map.insert({name_,result->cost_ms});
                broadcast(result);
            }
        }
    }
}

// ============================================================
// 多 batch 推理核心
// ============================================================
std::vector<std::shared_ptr<core::InferenceResultPacket>> DetectionInferNode::processBatch(
    const std::vector<std::shared_ptr<core::VideoFramePacket>>& frames) {

    int actual_batch = static_cast<int>(frames.size());
    std::vector<std::shared_ptr<core::InferenceResultPacket>> results;
    results.reserve(actual_batch);

    // 预先创建结果包
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
        LOG_DEBUG_FMT("[DetectionInfer] Mock inference: {} frames", actual_batch);
        return results;
    }

    try {
        // 1. 收集有效的 Mat 指针
        std::vector<cv::Mat*> valid_mats;
        std::vector<int> valid_indices;
        valid_mats.reserve(actual_batch);
        valid_indices.reserve(actual_batch);

        float scale_x[actual_batch];
        float scale_y[actual_batch];

        for (int b = 0; b < actual_batch; ++b) {
            if (!frames[b]->mat || frames[b]->mat->empty()) {
                LOG_ERROR_FMT("[DetectionInfer] Frame[{}] mat is null or empty", b);
                continue;
            }
            if (frames[b]->mat->type() != CV_32FC3) {
                LOG_ERROR_FMT("[DetectionInfer] Frame[{}] mat type {} != CV_32FC3({})", 
                             b, frames[b]->mat->type(), CV_32FC3);
                continue;
            }
            valid_mats.push_back(frames[b]->mat.get());
            valid_indices.push_back(b);

            // 计算每帧的缩放比例（基于 source_mat 原始尺寸）
            if (frames[b]->source_mat) {
                scale_x[b] = static_cast<float>(frames[b]->source_mat->cols) / INPUT_W;
                scale_y[b] = static_cast<float>(frames[b]->source_mat->rows) / INPUT_H;
            } else {
                scale_x[b] = 1.0f;
                scale_y[b] = 1.0f;
            }
        }

        int valid_batch = static_cast<int>(valid_mats.size());
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

        // 3. 多帧预处理 -> 一个连续的 GPU buffer
        preprocessBatch(valid_mats, static_cast<float*>(d_input_), valid_batch);
        cudaStreamSynchronize(stream_);

        // 设置 Tensor 地址
        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(boxes_name_.c_str(), d_boxes_);
        context_->setTensorAddress(scores_name_.c_str(), d_scores_);
        context_->setTensorAddress(classes_name_.c_str(), d_classes_);
        context_->setTensorAddress(batch_ids_name_.c_str(), d_batch_ids_);  // 新增
        context_->setTensorAddress(num_dets_name_.c_str(), d_num_dets_);

        // 执行推理
        auto t0 = std::chrono::high_resolution_clock::now();
        
        if (!context_->enqueueV3(stream_)) {
            LOG_ERROR_FMT("[DetectionInfer] enqueueV3 failed for batch={}", valid_batch);
            return results;
        }
        cudaStreamSynchronize(stream_);

        auto t1 = std::chrono::high_resolution_clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // 拷贝 det_num_dets 回 host（scalar）
        cudaMemcpyAsync(&h_num_dets_, d_num_dets_, sizeof(int64_t),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        int total_dets = static_cast<int>(h_num_dets_);
        LOG_INFO_FMT("[DetectionInfer] Total detections: {}", total_dets);

        if (total_dets <= 0) {
            LOG_DEBUG_FMT("[DetectionInfer] No detections in batch");
            return results;
        }

        // 限制上限，防止越界
        int max_total_dets = max_batch_size_.load() * MAX_DETS;
        if (total_dets > max_total_dets) {
            LOG_WARN_FMT("[DetectionInfer] Total detections {} exceeds buffer {}, clamping", 
                         total_dets, max_total_dets);
            total_dets = max_total_dets;
        }

        // 只拷贝实际需要的输出数据
        size_t actual_out_boxes_size = static_cast<size_t>(total_dets) * 4 * sizeof(float);
        size_t actual_out_scores_size = static_cast<size_t>(total_dets) * sizeof(float);
        size_t actual_out_classes_size = static_cast<size_t>(total_dets) * sizeof(int64_t);
        size_t actual_out_batch_ids_size = static_cast<size_t>(total_dets) * sizeof(int64_t);

        cudaMemcpyAsync(h_boxes_.data(), d_boxes_, actual_out_boxes_size,
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_scores_.data(), d_scores_, actual_out_scores_size,
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_classes_.data(), d_classes_, actual_out_classes_size,
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_batch_ids_.data(), d_batch_ids_, actual_out_batch_ids_size,
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 构建有效帧的 scale 数组（ONNX batch_id 0..valid_batch-1 对应 valid_indices[0..valid_batch-1]）
        std::vector<float> valid_scale_x(valid_batch);
        std::vector<float> valid_scale_y(valid_batch);
        for (int i = 0; i < valid_batch; ++i) {
            valid_scale_x[i] = scale_x[valid_indices[i]];
            valid_scale_y[i] = scale_y[valid_indices[i]];
        }

        // 后处理：按 det_batch_ids 分组
        auto all_detections = postprocessBatch(valid_batch, total_dets, 
                                                valid_scale_x.data(), valid_scale_y.data(), 0.25f);

        for (int i = 0; i < valid_batch; ++i) {
            int orig_idx = valid_indices[i];
            results[orig_idx]->detections = std::move(all_detections[i]);
            LOG_INFO_FMT("[DetectionInfer] Frame[{}]: source id:{} stream:{}  {} dets, ts={}ms",
                         orig_idx, results[orig_idx]->source_id, results[orig_idx]->stream_id,
                         results[orig_idx]->detections.size(), frames[orig_idx]->timestamp_ms);
        }

        LOG_INFO_FMT("[DetectionInfer] Batch done: {}/{} valid, infer={:.2f}ms",
                     valid_batch, actual_batch, infer_ms);

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] Batch inference exception: {}", e.what());
    }

    return results;
}

// ============================================================
// 多帧预处理
// ============================================================
void DetectionInferNode::preprocessBatch(const std::vector<cv::Mat*>& images, 
                                          float* gpu_buffer, int batch_size) {
    int hw = INPUT_H * INPUT_W;
    int batch_stride = 3 * hw;
    
    alignas(64) static thread_local std::vector<float> host_input;
    host_input.resize(batch_size * batch_stride);

    for (int b = 0; b < batch_size; ++b) {
        const cv::Mat& image = *images[b];
        CV_Assert(image.type() == CV_32FC3);
        CV_Assert(image.rows == INPUT_H && image.cols == INPUT_W);

        std::vector<cv::Mat> chs(3);
        cv::split(image, chs);

        float* batch_ptr = host_input.data() + b * batch_stride;
        memcpy(batch_ptr + 0 * hw, chs[0].data, hw * sizeof(float));
        memcpy(batch_ptr + 1 * hw, chs[1].data, hw * sizeof(float));
        memcpy(batch_ptr + 2 * hw, chs[2].data, hw * sizeof(float));
    }

    cudaMemcpyAsync(gpu_buffer, host_input.data(), 
                    batch_size * batch_stride * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
}

// ============================================================
// 多 batch 后处理（使用 det_batch_ids 按帧分组）
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
// 初始化引擎
// ============================================================
bool DetectionInferNode::initEngine(const std::string& engine_path) {
    try {
        // 1. 读取文件
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

        // 2. 创建 runtime
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
        if (!runtime) {
            LOG_ERROR_FMT("[DetectionInfer] createInferRuntime failed");
            return false;
        }

        // 3. 反序列化引擎
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
        if (!engine) {
            LOG_ERROR_FMT("[DetectionInfer] deserializeCudaEngine failed");
            delete runtime;
            return false;
        }

        // 4. 创建上下文
        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        if (!context) {
            LOG_ERROR_FMT("[DetectionInfer] createExecutionContext failed");
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
        LOG_INFO_FMT("[DetectionInfer] Engine has {} I/O tensors", nb_io);
        
        bool has_batch_ids = false;
        bool has_num_dets = false;
        for (int i = 0; i < nb_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            
            std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
            LOG_INFO_FMT("[DetectionInfer]  Tensor[{}]: {} ({}), shape=[{}]", 
                         i, name, mode_str, dims.nbDims);
            
            if (strcmp(name, batch_ids_name_.c_str()) == 0) {
                has_batch_ids = true;
                LOG_INFO_FMT("[DetectionInfer]    -> det_batch_ids tensor found");
            }
            if (strcmp(name, num_dets_name_.c_str()) == 0) {
                has_num_dets = true;
                LOG_INFO_FMT("[DetectionInfer]    -> det_num_dets tensor found");
            }
        }

        if (!has_batch_ids) {
            LOG_WARN_FMT("[DetectionInfer] Engine does not have '{}' output, batch grouping may fail", 
                         batch_ids_name_);
        }
        if (!has_num_dets) {
            LOG_WARN_FMT("[DetectionInfer] Engine does not have '{}' output, will use score scanning fallback", 
                         num_dets_name_);
        }

        // 计算缓冲区大小
        int max_batch = max_batch_size_.load();
        input_size_ = static_cast<size_t>(max_batch) * 3 * INPUT_H * INPUT_W * sizeof(float);
        out_boxes_size_ = static_cast<size_t>(max_batch) * MAX_DETS * 4 * sizeof(float);
        out_scores_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(float);
        out_classes_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);
        out_batch_ids_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);  // 新增
        out_num_dets_size_ = sizeof(int64_t);  // scalar

        // 分配 GPU 内存
        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_boxes_, out_boxes_size_);
        cudaMalloc(&d_scores_, out_scores_size_);
        cudaMalloc(&d_classes_, out_classes_size_);
        cudaMalloc(&d_batch_ids_, out_batch_ids_size_);  // 新增
        cudaMalloc(&d_num_dets_, out_num_dets_size_);

        // 预分配 CPU 缓冲区
        h_boxes_.resize(max_batch * MAX_DETS * 4);
        h_scores_.resize(max_batch * MAX_DETS);
        h_classes_.resize(max_batch * MAX_DETS);
        h_batch_ids_.resize(max_batch * MAX_DETS);  // 新增
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
