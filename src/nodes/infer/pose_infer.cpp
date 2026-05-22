#include "pose_infer.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "tensor_rt_logger.h"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>

namespace ai_stream {
namespace nodes {


PoseInferNode::PoseInferNode() : IInferNode("PoseInfer") {
    LOG_DEBUG_FMT("[PoseInfer] Constructor");
}

PoseInferNode::~PoseInferNode() {
    stop();
    if (d_input_) cudaFree(d_input_);
    if (d_output_) cudaFree(d_output_);
    if (stream_) cudaStreamDestroy(stream_);
    LOG_DEBUG_FMT("[PoseInfer] Destructor");
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
    max_batch_size_ = batch_size;
    queue_.setMaxSize(batch_size_ * 4);
    LOG_INFO_FMT("[PoseInfer] Set batch size: {}, queue size: {}", batch_size, queue_.getMaxSize());
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
    LOG_INFO_FMT("[PoseInfer] Started with max_batch={}", max_batch_size_.load());
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
    if (packet->type != core::PacketType::DECODED_FRAME) return;
    auto frame = std::dynamic_pointer_cast<core::VideoFramePacket>(packet);
    if (frame) {
        queue_.push(frame, std::chrono::milliseconds(10));
    }
}

void PoseInferNode::inferLoop() {
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
        LOG_DEBUG_FMT("[PoseInfer] Batch collected: {}/{}", actual_batch, max_batch_size_.load());

        auto t0 = std::chrono::high_resolution_clock::now();
        auto results = processBatch(batch_frames);
        auto t1 = std::chrono::high_resolution_clock::now();

        float batch_infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        LOG_INFO_FMT("[PoseInfer] Batch inference: {} frames, total={:.2f}ms, avg={:.2f}ms/frame",
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

std::vector<std::shared_ptr<core::InferenceResultPacket>> PoseInferNode::processBatch(
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
        // Mock 模式：填充假关键点
        for (int b = 0; b < actual_batch; ++b) {
            core::InferenceResultPacket::PoseResult pr;
            pr.person_score = 0.9f;
            for (int k = 0; k < NUM_KPTS; ++k) {
                //pr.keypoints[k] = {100.0f + k * 10, 100.0f + k * 5, 0.8f, true};
                pr.keypoints[k] = core::InferenceResultPacket::KeyPoint(100.0f + k * 10, 100.0f + k * 5, 0.8f, true);
            }
            pr.person_box = cv::Rect2f(100, 100, 200, 300);
            pr.matched_det_idx = 0;
            results[b]->pose_results.push_back(pr);
        }
        LOG_DEBUG_FMT("[PoseInfer] Mock inference: {} frames", actual_batch);
        return results;
    }

    try {
        std::vector<cv::Mat*> valid_mats;
        std::vector<int> valid_indices;
        std::vector<cv::Rect2f> crop_rois;
        std::vector<cv::Size> source_sizes;
        valid_mats.reserve(actual_batch);
        valid_indices.reserve(actual_batch);
        crop_rois.reserve(actual_batch);
        source_sizes.reserve(actual_batch);

        for (int b = 0; b < actual_batch; ++b) {
            if (!frames[b]->mat || frames[b]->mat->empty()) {
                LOG_ERROR_FMT("[PoseInfer] Frame[{}] mat is null or empty", b);
                continue;
            }
            if (frames[b]->mat->type() != CV_32FC3) {
                LOG_ERROR_FMT("[PoseInfer] Frame[{}] mat type {} != CV_32FC3({})", 
                             b, frames[b]->mat->type(), CV_32FC3);
                continue;
            }
            valid_mats.push_back(frames[b]->mat.get());
            valid_indices.push_back(b);
            source_sizes.emplace_back(
                frames[b]->source_mat ? frames[b]->source_mat->size() : frames[b]->mat->size());
            
            // 获取 crop_roi（如果有）
            if (frames[b]->crop_roi.width > 0 && frames[b]->crop_roi.height > 0) {
                crop_rois.push_back(frames[b]->crop_roi);
            } else {
                crop_rois.emplace_back(0, 0, 
                    static_cast<float>(frames[b]->mat->cols), 
                    static_cast<float>(frames[b]->mat->rows));
                if (frames[b]->source_mat) {
                    LOG_WARN_FMT("[PoseInfer] Frame[{}] has no crop_roi, fallback to full-image scaling", b);
                }
            }
        }

        int valid_batch = static_cast<int>(valid_mats.size());
        if (valid_batch == 0) {
            LOG_WARN_FMT("[PoseInfer] No valid frames in batch");
            return results;
        }

        // 设置动态输入
        nvinfer1::Dims4 input_dims(valid_batch, 3, INPUT_H, INPUT_W);
        if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
            LOG_ERROR_FMT("[PoseInfer] setInputShape failed for batch={}", valid_batch);
            return results;
        }

        preprocessBatch(valid_mats, static_cast<float*>(d_input_), valid_batch);
        cudaStreamSynchronize(stream_);

        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(output_name_.c_str(), d_output_);

        auto t0 = std::chrono::high_resolution_clock::now();
        if (!context_->enqueueV3(stream_)) {
            LOG_ERROR_FMT("[PoseInfer] enqueueV3 failed for batch={}", valid_batch);
            return results;
        }
        cudaStreamSynchronize(stream_);
        auto t1 = std::chrono::high_resolution_clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // 拷贝输出到 host
        int total_output_size = valid_batch * MAX_CANDIDATES * OUT_DIM;
        cudaMemcpyAsync(h_output_.data(), d_output_, 
                        valid_batch * MAX_CANDIDATES * OUT_DIM * sizeof(float),
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 后处理
        postprocessBatch(valid_batch, h_output_.data(), crop_rois, source_sizes, 
                         conf_thresh_, results);

        // 把结果放到正确的索引位置
        // postprocessBatch 直接操作了 results[valid_indices[i]]
        // 所以不需要额外移动

        LOG_INFO_FMT("[PoseInfer] Batch done: {}/{} valid, infer={:.2f}ms",
                     valid_batch, actual_batch, infer_ms);

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[PoseInfer] Batch inference exception: {}", e.what());
    }

    return results;
}

void PoseInferNode::preprocessBatch(const std::vector<cv::Mat*>& images, 
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

void PoseInferNode::postprocessBatch(
    int batch_size,
    const float* host_output,
    const std::vector<cv::Rect2f>& crop_rois,
    const std::vector<cv::Size>& source_sizes,
    float conf_thresh,
    std::vector<std::shared_ptr<core::InferenceResultPacket>>& results) {

    for (int b = 0; b < batch_size; ++b) {
        const float* frame_output = host_output + b * MAX_CANDIDATES * OUT_DIM;  // [N, 56]

        // 遍历所有候选，找 person score 最高的
        int best_idx = -1;
        float best_score = -1.0f;

        for (int i = 0; i < MAX_CANDIDATES; ++i) {
            const float* cand = frame_output + i * OUT_DIM;
            float raw_score = cand[4];
            float score = 1.0f / (1.0f + std::exp(-raw_score));  // sigmoid
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        if (best_idx < 0 || best_score < conf_thresh) {
            LOG_DEBUG_FMT("[PoseInfer] Frame[{}] no person detected (best_score={:.3f})", b, best_score);
            continue;
        }

        const float* best = frame_output + best_idx * OUT_DIM;
        float box_cx = best[0];
        float box_cy = best[1];
        float box_w  = best[2];
        float box_h  = best[3];

        // 关键点解码
        core::InferenceResultPacket::PoseResult pr;
        pr.person_score = best_score;
        pr.matched_det_idx = 0;  // 单目标场景，默认匹配第0个检测框

        // 坐标映射参数
        const cv::Rect2f& roi = crop_rois[b];
        bool has_crop = (roi.width > 0 && roi.height > 0 && 
                         (roi.x != 0 || roi.y != 0 || 
                          std::abs(roi.width - INPUT_W) > 1e-3f));
        
        float scale_x = 1.0f, scale_y = 1.0f;
        float offset_x = 0.0f, offset_y = 0.0f;

        if (has_crop) {
            scale_x = roi.width / static_cast<float>(INPUT_W);
            scale_y = roi.height / static_cast<float>(INPUT_H);
            offset_x = roi.x;
            offset_y = roi.y;
        } else {
            // 回退：按 source_mat 和 640 的比例（无偏移）
            scale_x = source_sizes[b].width / static_cast<float>(INPUT_W);
            scale_y = source_sizes[b].height / static_cast<float>(INPUT_H);
        }

        // person_box 映射回原图
        float x1 = (box_cx - box_w * 0.5f) * scale_x + offset_x;
        float y1 = (box_cy - box_h * 0.5f) * scale_y + offset_y;
        float w  = box_w * scale_x;
        float h  = box_h * scale_y;
        pr.person_box = cv::Rect2f(x1, y1, w, h);

        // 17 个关键点
        for (int k = 0; k < NUM_KPTS; ++k) {
            float kx = best[5 + k * KPT_DIMS + 0];
            float ky = best[5 + k * KPT_DIMS + 1];
            float kconf_raw = best[5 + k * KPT_DIMS + 2];
            float kconf = 1.0f / (1.0f + std::exp(-kconf_raw));  // sigmoid visibility

            float orig_kx = kx * scale_x + offset_x;
            float orig_ky = ky * scale_y + offset_y;

            // 填充到用户的 KeyPoint 结构（假设 KeyPoint 有 x, y, confidence, visible）
            pr.keypoints[k].x = orig_kx;
            pr.keypoints[k].y = orig_ky;
            // 如果用户的 KeyPoint 有 confidence 和 visible 字段：
            // pr.keypoints[k].confidence = kconf;
            // pr.keypoints[k].visible = (kconf > kpt_conf_thresh_);
        }

        results[b]->pose_results.push_back(pr);
        LOG_INFO_FMT("[PoseInfer] Frame[{}] pose detected, score={:.3f}, kpts={}/{}, box=({:.1f},{:.1f},{:.1f},{:.1f})",
                     b, best_score, 
                     std::count_if(pr.keypoints.begin(), pr.keypoints.end(), 
                                  [this](const auto& k){ return k.confidence > kpt_conf_thresh_; }),
                     NUM_KPTS, x1, y1, w, h);
    }
}

bool PoseInferNode::initEngine(const std::string& engine_path) {
    try {
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

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
        if (!runtime) {
            LOG_ERROR_FMT("[PoseInfer] createInferRuntime failed");
            return false;
        }
        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
        if (!engine) {
            LOG_ERROR_FMT("[PoseInfer] deserializeCudaEngine failed");
            delete runtime;
            return false;
        }
        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        if (!context) {
            LOG_ERROR_FMT("[PoseInfer] createExecutionContext failed");
            delete engine;
            delete runtime;
            return false;
        }

        runtime_.reset(runtime);
        engine_.reset(engine);
        context_.reset(context);
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

        int max_batch = max_batch_size_.load();
        input_size_ = static_cast<size_t>(max_batch) * 3 * INPUT_H * INPUT_W * sizeof(float);
        output_size_ = static_cast<size_t>(max_batch) * MAX_CANDIDATES * OUT_DIM * sizeof(float);

        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_output_, output_size_);
        h_output_.resize(max_batch * MAX_CANDIDATES * OUT_DIM);

        LOG_INFO_FMT("[PoseInfer] Engine loaded: {} (max_batch={}, max_candidates={})",
                     engine_path, max_batch, MAX_CANDIDATES);
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[PoseInfer] initEngine exception: {}", e.what());
        return false;
    }
}

REGISTER_NODE("pose_infer", PoseInferNode)

} // namespace nodes
} // namespace ai_stream