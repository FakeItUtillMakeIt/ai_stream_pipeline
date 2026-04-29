// src/nodes/infer/detection_infer.cpp
#include "detection_infer.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

#include <NvInfer.h>

namespace ai_stream {
namespace nodes {

// ============================================================
// TensorRT Logger
// ============================================================
class TensorRTLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
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
            default:
                LOG_DEBUG_FMT("[TensorRT] {}", msg);
                break;
        }
    }
};

static TensorRTLogger g_logger;

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
    queue_.setMaxSize(batch_size_ * 4);  // 增大队列以容纳更多帧
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
    if (!running_) return;
    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::dynamic_pointer_cast<core::VideoFramePacket>(packet);
    if (frame) {
        queue_.push(frame, std::chrono::milliseconds(10));
    }
}

// ============================================================
// 推理主循环：收集 batch 并执行推理
// ============================================================
void DetectionInferNode::inferLoop() {
    while (running_) {
        std::vector<std::shared_ptr<core::VideoFramePacket>> batch_frames;
        batch_frames.reserve(max_batch_size_);

        // 1. 等待第一帧（阻塞等待）
        std::shared_ptr<core::VideoFramePacket> first_frame;
        if (!queue_.pop(first_frame, std::chrono::milliseconds(100))) {
            continue;
        }
        batch_frames.push_back(first_frame);

        auto batch_start_time = std::chrono::high_resolution_clock::now();

        // 2. 尝试收集更多帧凑齐 batch（带超时）
        while (static_cast<int>(batch_frames.size()) < max_batch_size_) {
            std::shared_ptr<core::VideoFramePacket> frame;
            auto elapsed = std::chrono::high_resolution_clock::now() - batch_start_time;
            auto remaining = batch_timeout_ms_ - std::chrono::duration_cast<std::chrono::milliseconds>(elapsed);
            
            if (remaining.count() <= 0) break;  // 超时，不再等待

            if (queue_.pop(frame, remaining)) {
                batch_frames.push_back(frame);
            } else {
                break;  // 队列为空或超时
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
        result->timestamp_ms = frames[b]->timestamp_ms;
        result->source_frame = frames[b];
        results.push_back(result);
    }

    if (!engine_ || !context_) {
        // Mock 模式：为每帧生成模拟结果
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

    // ========== TensorRT 10 多 Batch 推理 ==========
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

        // 2. 设置动态输入形状（TensorRT 10 API）
        nvinfer1::Dims4 input_dims(valid_batch, 3, INPUT_H, INPUT_W);
        if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
            LOG_ERROR_FMT("[DetectionInfer] setInputShape failed for batch={}", valid_batch);
            return results;
        }

        // 3. 多帧预处理 -> 一个连续的 GPU buffer
        preprocessBatch(valid_mats, static_cast<float*>(d_input_), valid_batch);
        cudaStreamSynchronize(stream_);

        // 4. 设置输入输出地址（TensorRT 10: enqueueV3）
        // 注意：输出缓冲区大小需要根据实际 batch 调整
        size_t actual_out_boxes_size = static_cast<size_t>(valid_batch) * MAX_DETS * 4 * sizeof(float);
        size_t actual_out_scores_size = static_cast<size_t>(valid_batch) * MAX_DETS * sizeof(float);
        size_t actual_out_classes_size = static_cast<size_t>(valid_batch) * MAX_DETS * sizeof(int64_t);

        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(boxes_name_.c_str(), d_boxes_);
        context_->setTensorAddress(scores_name_.c_str(), d_scores_);
        context_->setTensorAddress(classes_name_.c_str(), d_classes_);

        // 5. 执行推理
        auto t0 = std::chrono::high_resolution_clock::now();
        
        if (!context_->enqueueV3(stream_)) {
            LOG_ERROR_FMT("[DetectionInfer] enqueueV3 failed for batch={}", valid_batch);
            return results;
        }
        cudaStreamSynchronize(stream_);

        auto t1 = std::chrono::high_resolution_clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // 6. 获取输出形状
        nvinfer1::Dims out_boxes_dims = context_->getTensorShape(boxes_name_.c_str());
        nvinfer1::Dims out_scores_dims = context_->getTensorShape(scores_name_.c_str());
        nvinfer1::Dims out_classes_dims = context_->getTensorShape(classes_name_.c_str());
        LOG_DEBUG_FMT("[DetectionInfer] Output shapes: boxes=[{}] scores=[{}] classes=[{}]", 
                      out_boxes_dims.d[0], out_scores_dims.d[0], out_classes_dims.d[0]);

        // 7. 拷贝输出回 Host（只拷贝实际 batch 的数据）
        cudaMemcpyAsync(h_boxes_.data(), d_boxes_, actual_out_boxes_size,
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_scores_.data(), d_scores_, actual_out_scores_size,
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_classes_.data(), d_classes_, actual_out_classes_size,
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 8. 解析每帧的检测数量
        int num_dets_per_batch[valid_batch];
        for (int b = 0; b < valid_batch; ++b) {
            num_dets_per_batch[b] = 0;
            for (int i = 0; i < MAX_DETS; ++i) {
                float score = h_scores_[b * MAX_DETS + i];
                if (score < 0.001f || std::isnan(score)) break;
                num_dets_per_batch[b]++;
            }
        }

        // 9. 后处理：解析每个 batch 的结果
        auto all_detections = postprocessBatch(valid_batch, num_dets_per_batch, 
                                                scale_x, scale_y, 0.25f);

        // 10. 将结果回填到对应帧
        for (int i = 0; i < valid_batch; ++i) {
            int orig_idx = valid_indices[i];
            results[orig_idx]->detections = std::move(all_detections[i]);
            LOG_INFO_FMT("[DetectionInfer] Frame[{}]: {} dets, ts={}ms",
                         orig_idx, results[orig_idx]->detections.size(), 
                         frames[orig_idx]->timestamp_ms);
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
    
    // 使用 pinned memory 或异步拷贝优化
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

    // 一次性拷贝整个 batch 到 GPU
    cudaMemcpyAsync(gpu_buffer, host_input.data(), 
                    batch_size * batch_stride * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
}

// ============================================================
// 多 batch 后处理
// ============================================================
std::vector<std::vector<core::InferenceResultPacket::BBox>> DetectionInferNode::postprocessBatch(
    int batch_size, int num_dets_per_batch[], float scale_x[], float scale_y[], float conf_thresh) {
    
    std::vector<std::vector<core::InferenceResultPacket::BBox>> all_detections;
    all_detections.reserve(batch_size);

    for (int b = 0; b < batch_size; ++b) {
        std::vector<core::InferenceResultPacket::BBox> detections;
        int num_dets = num_dets_per_batch[b];
        detections.reserve(num_dets);

        int batch_offset = b * MAX_DETS;

        for (int i = 0; i < num_dets; ++i) {
            float score = h_scores_[batch_offset + i];
            if (score < conf_thresh) continue;

            float cx = h_boxes_[(batch_offset + i) * 4 + 0];
            float cy = h_boxes_[(batch_offset + i) * 4 + 1];
            float w  = h_boxes_[(batch_offset + i) * 4 + 2];
            float h  = h_boxes_[(batch_offset + i) * 4 + 3];

            core::InferenceResultPacket::BBox box;
            box.x = static_cast<int>((cx - w / 2.0f) * scale_x[b]);
            box.y = static_cast<int>((cy - h / 2.0f) * scale_y[b]);
            box.w = static_cast<int>(w * scale_x[b]);
            box.h = static_cast<int>(h * scale_y[b]);
            box.confidence = score;
            box.class_id = static_cast<int>(h_classes_[batch_offset + i]);
            
            if (box.class_id >= 0 && box.class_id < static_cast<int>(class_names_.size())) {
                box.class_name = class_names_[box.class_id];
            } else {
                box.class_name = "unknown";
            }

            detections.push_back(box);
        }
        all_detections.push_back(std::move(detections));
    }
    return all_detections;
}

// ============================================================
// 初始化引擎（TensorRT 10 API）
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

        // 1. 显存检查
        size_t free_mem = 0, total_mem = 0;
        cudaMemGetInfo(&free_mem, &total_mem);

        // 2. 用最大 batch 预热（触发内部分配）
        cudaStream_t warmup_stream;
        cudaStreamCreate(&warmup_stream);

        // 分配临时缓冲区
        void *d_tmp_input, *d_tmp_boxes, *d_tmp_scores, *d_tmp_classes;
        cudaMalloc(&d_tmp_input, max_batch_size_ * 3 * INPUT_H * INPUT_W * sizeof(float));
        cudaMalloc(&d_tmp_boxes, max_batch_size_ * MAX_DETS * 4 * sizeof(float));
        cudaMalloc(&d_tmp_scores, max_batch_size_ * MAX_DETS * sizeof(float));
        cudaMalloc(&d_tmp_classes, max_batch_size_ * MAX_DETS * sizeof(int64_t));

        // 绑定并预热
        nvinfer1::Dims4 warmup_dims(max_batch_size_, 3, INPUT_H, INPUT_W);
        context->setInputShape(input_name_.c_str(), warmup_dims);
        context->setTensorAddress(input_name_.c_str(), d_tmp_input);
        context->setTensorAddress(boxes_name_.c_str(), d_tmp_boxes);
        context->setTensorAddress(scores_name_.c_str(), d_tmp_scores);
        context->setTensorAddress(classes_name_.c_str(), d_tmp_classes);

        bool warmup_ok = context->enqueueV3(warmup_stream);
        cudaStreamSynchronize(warmup_stream);

        if (!warmup_ok) {
            LOG_ERROR_FMT("[DetectionInfer] Warmup failed - insufficient GPU memory for batch={}", 
                        max_batch_size_.load());
            // 降级到 batch=1
            max_batch_size_ = 1;
            batch_size_ = 1;
        }

        // 清理临时资源
        cudaFree(d_tmp_input); cudaFree(d_tmp_boxes); 
        cudaFree(d_tmp_scores); cudaFree(d_tmp_classes);
        cudaStreamDestroy(warmup_stream);

        // 5. 保存资源
        runtime_.reset(runtime);
        engine_.reset(engine);
        context_.reset(context);

        // 6. 创建 CUDA Stream
        cudaStreamCreate(&stream_);

        // 7. 检查引擎是否支持动态 batch
        nvinfer1::Dims profile_dims = engine_->getProfileShape(input_name_.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        int max_engine_batch = profile_dims.d[0];
        LOG_INFO_FMT("[DetectionInfer] Engine max batch size: {}", max_engine_batch);

        // 限制 batch_size_ 不超过引擎支持的最大值
        if (batch_size_ > max_engine_batch) {
            LOG_WARN_FMT("[DetectionInfer] Requested batch_size({}) > engine max({}), limiting to {}", 
                        batch_size_, max_engine_batch, max_engine_batch);
            batch_size_ = max_engine_batch;
            max_batch_size_ = max_engine_batch;
        }

        // 8. 打印 Tensor 信息（TensorRT 10 API）
        int nb_io = engine_->getNbIOTensors();
        LOG_INFO_FMT("[DetectionInfer] Engine has {} I/O tensors", nb_io);
        
        for (int i = 0; i < nb_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            
            std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
            LOG_INFO_FMT("[DetectionInfer]  Tensor[{}]: {} ({}), shape=[{}]", 
                         i, name, mode_str, dims.nbDims);
            
            // 验证名称
            if (strcmp(name, input_name_.c_str()) == 0) {
                LOG_INFO_FMT("[DetectionInfer]    -> Input tensor confirmed");
            } else if (strcmp(name, boxes_name_.c_str()) == 0 ||
                       strcmp(name, scores_name_.c_str()) == 0 ||
                       strcmp(name, classes_name_.c_str()) == 0) {
                LOG_INFO_FMT("[DetectionInfer]    -> Output tensor confirmed");
            }
        }

        // 9. 计算缓冲区大小（按最大 batch size 分配）
        int max_batch = max_batch_size_.load();
        input_size_ = static_cast<size_t>(max_batch) * 3 * INPUT_H * INPUT_W * sizeof(float);
        out_boxes_size_ = static_cast<size_t>(max_batch) * MAX_DETS * 4 * sizeof(float);
        out_scores_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(float);
        out_classes_size_ = static_cast<size_t>(max_batch) * MAX_DETS * sizeof(int64_t);

        // 10. 分配 GPU 内存
        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_boxes_, out_boxes_size_);
        cudaMalloc(&d_scores_, out_scores_size_);
        cudaMalloc(&d_classes_, out_classes_size_);

        // 11. 预分配 CPU 缓冲区
        h_boxes_.resize(max_batch * MAX_DETS * 4);
        h_scores_.resize(max_batch * MAX_DETS);
        h_classes_.resize(max_batch * MAX_DETS);

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