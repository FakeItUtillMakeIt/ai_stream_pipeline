// src/nodes/infer/cuda_pose_infer.cu
#include "cuda_pose_infer.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "tensor_rt_logger.h"
#include "utils/cuda_check.h"

#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
#include <cmath>

#include <NvInfer.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ai_stream {
namespace nodes {


// ============================================================
// CUDA Kernel：融合 Crop + Bilinear Resize + Normalize + HWC→NCHW
// ============================================================
namespace {

__global__ void cropResizeNormalizeKernel(
    const unsigned char* __restrict__ src,
    int src_w, int src_h, size_t src_pitch,
    float* __restrict__ dst,
    const float* __restrict__ boxes,
    int num_persons,
    int dst_w, int dst_h,
    float mean_b, float mean_g, float mean_r,
    float std_b, float std_g, float std_r)
{
    int pw = blockIdx.x * blockDim.x + threadIdx.x;
    int ph = blockIdx.y * blockDim.y + threadIdx.y;
    int p  = blockIdx.z;

    if (p >= num_persons || pw >= dst_w || ph >= dst_h) {
        return;
    }

    // 读取检测框（原图坐标）—— 改名避免 Bessel 函数冲突
    float box_x1 = boxes[p * 4 + 0];
    float box_y1 = boxes[p * 4 + 1];
    float box_x2 = boxes[p * 4 + 2];
    float box_y2 = boxes[p * 4 + 3];
    float crop_w = box_x2 - box_x1;
    float crop_h = box_y2 - box_y1;

    // 反向映射：输出像素中心 → 原图采样点
    float scale_x = crop_w / dst_w;
    float scale_y = crop_h / dst_h;
    float src_x = box_x1 + (pw + 0.5f) * scale_x - 0.5f;
    float src_y = box_y1 + (ph + 0.5f) * scale_y - 0.5f;

    // Clamp 到有效范围
    src_x = fmaxf(0.0f, fminf(src_x, src_w - 1.0f));
    src_y = fmaxf(0.0f, fminf(src_y, src_h - 1.0f));

    int x0 = static_cast<int>(floorf(src_x));
    int y0 = static_cast<int>(floorf(src_y));

    // 用条件运算符替代 min/max 宏，彻底避免宏展开陷阱
    int x0p1 = x0 + 1;
    int y0p1 = y0 + 1;
    int x1_i = (x0p1 < src_w) ? x0p1 : (src_w - 1);
    int y1_i = (y0p1 < src_h) ? y0p1 : (src_h - 1);

    float dx = src_x - x0;
    float dy = src_y - y0;
    float w00 = (1.0f - dx) * (1.0f - dy);
    float w01 = dx * (1.0f - dy);
    float w10 = (1.0f - dx) * dy;
    float w11 = dx * dy;

    const unsigned char* row0 = src + y0 * src_pitch;
    const unsigned char* row1 = src + y1_i * src_pitch;

    // BGR 双线性采样
    float b = w00 * row0[x0 * 3 + 0] + w01 * row0[x1_i * 3 + 0]
            + w10 * row1[x0 * 3 + 0] + w11 * row1[x1_i * 3 + 0];
    float g = w00 * row0[x0 * 3 + 1] + w01 * row0[x1_i * 3 + 1]
            + w10 * row1[x0 * 3 + 1] + w11 * row1[x1_i * 3 + 1];
    float r = w00 * row0[x0 * 3 + 2] + w01 * row0[x1_i * 3 + 2]
            + w10 * row1[x0 * 3 + 2] + w11 * row1[x1_i * 3 + 2];

    // Normalize
    b = (b / 255.0f - mean_b) / std_b;
    g = (g / 255.0f - mean_g) / std_g;
    r = (r / 255.0f - mean_r) / std_r;

    // 写入 NCHW
    int hw = dst_w * dst_h;
    int base = p * 3 * hw + ph * dst_w + pw;
    dst[base + 0 * hw] = b;
    dst[base + 1 * hw] = g;
    dst[base + 2 * hw] = r;
}

} // anonymous namespace

// ============================================================
// CudaPoseInferNode
// ============================================================

CudaPoseInferNode::CudaPoseInferNode() : IInferNode("CudaPoseInfer") {
    LOG_INFO_FMT("[CudaPoseInfer] Constructor");
    d_input_ = nullptr;
    d_output_ = nullptr;
    d_source_img_ = nullptr;
    d_boxes_ = nullptr;
    stream_ = nullptr;
    source_buffer_size_ = 0;
}

CudaPoseInferNode::~CudaPoseInferNode() {
    stop();

    if (d_input_) { cudaFree(d_input_); d_input_ = nullptr; }
    if (d_output_) { cudaFree(d_output_); d_output_ = nullptr; }
    if (d_source_img_) { cudaFree(d_source_img_); d_source_img_ = nullptr; }
    if (d_boxes_) { cudaFree(d_boxes_); d_boxes_ = nullptr; }
    if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }

    LOG_INFO_FMT("[CudaPoseInfer] Destructor");
}

bool CudaPoseInferNode::loadModel(const std::string& model_path) {
    LOG_INFO_FMT("[CudaPoseInfer] Loading model from: {}", model_path);
    std::ifstream file(model_path, std::ios::binary);
    if (!file.good()) {
        LOG_ERROR_FMT("[CudaPoseInfer] Model file not found: {}", model_path);
        return false;
    }
    return initEngine(model_path);
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
    return {input_width_, input_height_};
}

std::vector<std::string> CudaPoseInferNode::getClassNames() const {
    return class_names_;
}

bool CudaPoseInferNode::start() {
    if (!engine_) {
        LOG_WARN_FMT("[CudaPoseInfer] No model loaded, will use mock inference");
    }
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
        stop();
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

    // 4. 上传检测框坐标到 GPU
    std::vector<float> h_boxes(static_cast<size_t>(num_persons) * 4);
    for (int i = 0; i < num_persons; ++i) {
        const auto& det = packet->detections[person_indices[i]];
        h_boxes[i * 4 + 0] = det.x;
        h_boxes[i * 4 + 1] = det.y;
        h_boxes[i * 4 + 2] = det.x + det.w;
        h_boxes[i * 4 + 3] = det.y + det.h;
    }
    CUDA_CHECK(cudaMemcpyAsync(d_boxes_, h_boxes.data(),
                               num_persons * 4 * sizeof(float),
                               cudaMemcpyHostToDevice, stream_));

    // 5. 设置动态 batch
    nvinfer1::Dims4 input_dims(num_persons, 3, INPUT_H, INPUT_W);
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
        LOG_ERROR_FMT("[CudaPoseInfer] setInputShape failed for batch={}", num_persons);
        return;
    }

    // 6. 启动融合 CUDA Kernel 直接生成模型输入（NCHW）
    dim3 block_size(16, 16, 1);
    dim3 grid_size((INPUT_W + block_size.x - 1) / block_size.x,
                   (INPUT_H + block_size.y - 1) / block_size.y,
                   num_persons);

    cropResizeNormalizeKernel<<<grid_size, block_size, 0, stream_>>>(
        static_cast<const unsigned char*>(d_source_img_),
        src_w, src_h, src_pitch,
        static_cast<float*>(d_input_),
        d_boxes_, num_persons,
        INPUT_W, INPUT_H,
        0.0f, 0.0f, 0.0f,   // mean BGR (默认仅 /255)
        1.0f, 1.0f, 1.0f    // std BGR
    );

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) {
        LOG_ERROR_FMT("[CudaPoseInfer] Preprocess kernel failed: {}", cudaGetErrorString(kernel_err));
        return;
    }

    // 7. 设置 Tensor 地址并执行推理
    size_t output_bytes = static_cast<size_t>(num_persons) * NUM_CANDIDATES * POSE_DIM * sizeof(float);

    context_->setTensorAddress(input_name_.c_str(), d_input_);
    context_->setTensorAddress(output_name_.c_str(), d_output_);

    auto t0 = std::chrono::high_resolution_clock::now();
    if (!context_->enqueueV3(stream_)) {
        LOG_ERROR_FMT("[CudaPoseInfer] enqueueV3 failed for batch={}", num_persons);
        return;
    }
    cudaStreamSynchronize(stream_);
    auto t1 = std::chrono::high_resolution_clock::now();

    float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
    LOG_INFO_FMT("[CudaPoseInfer] TensorRT enqueue: {} persons, {:.2f}ms", num_persons, infer_ms);

    // 8. 拷贝输出回 host
    h_output_.resize(static_cast<size_t>(num_persons) * NUM_CANDIDATES * POSE_DIM);
    CUDA_CHECK(cudaMemcpyAsync(h_output_.data(), d_output_, output_bytes,
                               cudaMemcpyDeviceToHost, stream_));
    cudaStreamSynchronize(stream_);

    // 9. 后处理：解码关键点
    postprocessFrame(packet, person_indices, num_persons, h_output_.data());
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

// ============================================================
// 后处理：每人体框独立解码关键点
// ============================================================
void CudaPoseInferNode::postprocessFrame(
    std::shared_ptr<core::InferenceResultPacket> packet,
    const std::vector<int>& person_indices,
    int num_persons,
    float* output_host) {

    packet->pose_results.clear();
    packet->pose_results.reserve(num_persons);

    for (int p = 0; p < num_persons; ++p) {
        int det_idx = person_indices[p];
        auto& det = packet->detections[det_idx];

        float* person_output = output_host + p * NUM_CANDIDATES * POSE_DIM;

        // 遍历 8400 候选，找 person score 最高的
        float best_score = -1.0f;
        int best_idx = 0;

        for (int i = 0; i < NUM_CANDIDATES; ++i) {
            float score = person_output[i * POSE_DIM + 4];
            score = 1.0f / (1.0f + std::exp(-score));
            if (score > best_score) {
                best_score = score;
                best_idx = i;
            }
        }

        if (best_score < conf_thresh_) {
            LOG_INFO_FMT("[CudaPoseInfer] Person {} best score {:.3f} below threshold {}, skipping",
                          det_idx, best_score, conf_thresh_);
            continue;
        }

        float* best_pred = person_output + best_idx * POSE_DIM;

        core::InferenceResultPacket::PoseResult pose;
        pose.person_score = best_score;
        pose.matched_det_idx = det_idx;

        float scale_x = det.w / static_cast<float>(INPUT_W);
        float scale_y = det.h / static_cast<float>(INPUT_H);

        for (int k = 0; k < NUM_KEYPOINTS; ++k) {
            float kx = best_pred[5 + k * 3 + 0];
            float ky = best_pred[5 + k * 3 + 1];
            float kconf = best_pred[5 + k * 3 + 2];

            kconf = 1.0f / (1.0f + std::exp(-kconf));

            float orig_kx = kx * scale_x + det.x;
            float orig_ky = ky * scale_y + det.y;

            pose.keypoints[k] = {
                orig_kx,
                orig_ky,
                kconf,
                kconf > kpt_conf_thresh_
            };

            det.has_keypoints = true;
            det.keypoints[k] = pose.keypoints[k];
            det.keypoints_conf = kconf;
        }

        pose.person_box = cv::Rect2f(det.x, det.y, det.w, det.h);
        packet->pose_results.push_back(pose);

        LOG_INFO_FMT("[CudaPoseInfer] Person {}: score={:.3f}, kpts_visible={}/17",
                      det_idx, best_score,
                      std::count_if(pose.keypoints.begin(), pose.keypoints.end(),
                                    [](const auto& k){ return k.visible; }));
    }
}

// ============================================================
// 初始化引擎
// ============================================================
bool CudaPoseInferNode::initEngine(const std::string& engine_path) {
    try {
        std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            LOG_ERROR_FMT("[CudaPoseInfer] Failed to open engine: {}", engine_path);
            return false;
        }

        size_t size = file.tellg();
        file.seekg(0, std::ios::beg);
        std::vector<char> buffer(size);
        if (!file.read(buffer.data(), size)) {
            LOG_ERROR_FMT("[CudaPoseInfer] Failed to read engine");
            return false;
        }

        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(g_logger);
        if (!runtime) {
            LOG_ERROR_FMT("[CudaPoseInfer] createInferRuntime failed");
            return false;
        }

        nvinfer1::ICudaEngine* engine = runtime->deserializeCudaEngine(buffer.data(), size);
        if (!engine) {
            LOG_ERROR_FMT("[CudaPoseInfer] deserializeCudaEngine failed");
            delete runtime;
            return false;
        }

        nvinfer1::IExecutionContext* context = engine->createExecutionContext();
        if (!context) {
            LOG_ERROR_FMT("[CudaPoseInfer] createExecutionContext failed");
            delete engine;
            delete runtime;
            return false;
        }

        runtime_.reset(runtime);
        engine_.reset(engine);
        context_.reset(context);

        cudaStreamCreate(&stream_);

        int nb_io = engine_->getNbIOTensors();
        LOG_INFO_FMT("[CudaPoseInfer] Engine has {} I/O tensors", nb_io);
        for (int i = 0; i < nb_io; ++i) {
            const char* name = engine_->getIOTensorName(i);
            nvinfer1::TensorIOMode mode = engine_->getTensorIOMode(name);
            nvinfer1::Dims dims = engine_->getTensorShape(name);
            std::string mode_str = (mode == nvinfer1::TensorIOMode::kINPUT) ? "INPUT" : "OUTPUT";
            LOG_INFO_FMT("[CudaPoseInfer]  Tensor[{}]: {} ({}), shape=[{}]",
                         i, name, mode_str, dims.nbDims);
        }

        int max_batch = batch_size_;
        input_size_ = static_cast<size_t>(max_batch) * 3 * INPUT_H * INPUT_W * sizeof(float);
        output_size_ = static_cast<size_t>(max_batch) * NUM_CANDIDATES * POSE_DIM * sizeof(float);

        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_output_, output_size_);
        cudaMalloc(&d_boxes_, max_batch * 4 * sizeof(float));

        h_output_.resize(static_cast<size_t>(max_batch) * NUM_CANDIDATES * POSE_DIM);

        LOG_INFO_FMT("[CudaPoseInfer] Engine loaded: {} (max_persons={}, input={}x{}, candidates={})",
                     engine_path, max_batch, INPUT_W, INPUT_H, NUM_CANDIDATES);
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[CudaPoseInfer] initEngine exception: {}", e.what());
        return false;
    }
}

REGISTER_NODE("cuda_pose_infer", CudaPoseInferNode)

} // namespace nodes
} // namespace ai_stream