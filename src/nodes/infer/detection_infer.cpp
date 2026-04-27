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
    LOG_INFO_FMT("[DetectionInfer] Set batch size: {}", batch_size);
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
    LOG_INFO_FMT("[DetectionInfer] Started");
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
    if (!running_) return;
    if (packet->type != core::PacketType::DECODED_FRAME) return;

    auto frame = std::dynamic_pointer_cast<core::VideoFramePacket>(packet);
    if (frame) {
        queue_.push(frame, std::chrono::milliseconds(10));
    }
}

void DetectionInferNode::inferLoop() {
    while (running_) {
        std::shared_ptr<core::VideoFramePacket> frame;
        if (!queue_.pop(frame, std::chrono::milliseconds(100))) {
            continue;
        }
        auto result = processFrame(frame);
        if (result) {
            broadcast(result);
        }
    }
}

// ============================================================
// 预处理
// ============================================================
void DetectionInferNode::preprocess(const cv::Mat& image, float* gpu_buffer) {
    

    CV_Assert(image.type() == CV_32FC3);
    CV_Assert(image.rows == INPUT_H && image.cols == INPUT_W);
    
    std::vector<cv::Mat> chs(3);
    cv::split(image, chs);

    int hw = INPUT_H * INPUT_W;
    alignas(64) static thread_local std::vector<float> host_input;
    host_input.resize(3 * hw);
    memcpy(host_input.data() + 0 * hw, chs[0].data, hw * sizeof(float));
    memcpy(host_input.data() + 1 * hw, chs[1].data, hw * sizeof(float));
    memcpy(host_input.data() + 2 * hw, chs[2].data, hw * sizeof(float));

    cudaMemcpyAsync(gpu_buffer, host_input.data(), 3 * hw * sizeof(float),
                    cudaMemcpyHostToDevice, stream_);
}

// ============================================================
// 后处理
// ============================================================
std::vector<core::InferenceResultPacket::BBox> DetectionInferNode::postprocess(
    int num_dets, float scale_x, float scale_y, float conf_thresh) {
    
    std::vector<core::InferenceResultPacket::BBox> detections;
    detections.reserve(num_dets);

    for (int i = 0; i < num_dets; ++i) {
        float score = h_scores_[i];
        if (score < conf_thresh) continue;

        float cx = h_boxes_[i * 4 + 0];
        float cy = h_boxes_[i * 4 + 1];
        float w  = h_boxes_[i * 4 + 2];
        float h  = h_boxes_[i * 4 + 3];

        core::InferenceResultPacket::BBox box;
        box.x = static_cast<int>((cx - w / 2.0f) * scale_x);
        box.y = static_cast<int>((cy - h / 2.0f) * scale_y);
        box.w = static_cast<int>(w * scale_x);
        box.h = static_cast<int>(h * scale_y);
        box.confidence = score;
        box.class_id = static_cast<int>(h_classes_[i]);
        
        if (box.class_id >= 0 && box.class_id < static_cast<int>(class_names_.size())) {
            box.class_name = class_names_[box.class_id];
        } else {
            box.class_name = "unknown";
        }

        detections.push_back(box);
    }
    return detections;
}

// ============================================================
// 单帧推理
// ============================================================
std::shared_ptr<core::InferenceResultPacket> DetectionInferNode::processFrame(
    std::shared_ptr<core::VideoFramePacket> frame) {

    if (frame->mat) {
        LOG_DEBUG_FMT("[DetectionInfer] Frame mat info: type={}, rows={}, cols={}, channels={}",
                      frame->mat->type(), frame->mat->rows, frame->mat->cols, frame->mat->channels());
        // OpenCV type 编码: CV_8UC3=16, CV_32FC3=21
        // 打印 type 的数值帮助诊断
        std::cout << "[DEBUG] mat type value: " << frame->mat->type() 
                  << " (expected CV_32FC3=" << CV_32FC3 << ")" << std::endl;
    } else {
        LOG_ERROR_FMT("[DetectionInfer] Frame mat is null");
    }

    auto result = std::make_shared<core::InferenceResultPacket>();
    result->stream_id = frame->stream_id;
    result->timestamp_ms = frame->timestamp_ms;
    result->source_frame = frame;

    if (!engine_ || !context_) {
        // Mock 模式
        static int frame_count = 0;
        frame_count++;
        for (int i = 0; i < 3; ++i) {
            core::InferenceResultPacket::BBox box;
            box.x = 100 + (frame_count % 200) + i * 50;
            box.y = 100 + ((frame_count / 2) % 200) + i * 30;
            box.w = 150 + (frame_count % 100);
            box.h = 200 + ((frame_count / 3) % 100);
            box.confidence = 0.7f + (frame_count % 30) / 100.0f;
            box.class_id = (frame_count + i) % class_names_.size();
            box.class_name = class_names_[box.class_id];
            result->detections.push_back(box);
        }
        LOG_DEBUG_FMT("[DetectionInfer] Mock inference: {} detections", result->detections.size());
        return result;
    }

    // ========== TensorRT 10 推理 ==========
    try {
        if (!frame->mat || frame->mat->empty()) {
            LOG_ERROR_FMT("[DetectionInfer] Frame mat is null or empty");
            return result;
        }
        // 1. 设置输入形状（TensorRT 10 API）
        nvinfer1::Dims4 input_dims(1, 3, INPUT_H, INPUT_W);
        if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
            LOG_ERROR_FMT("[DetectionInfer] setInputShape failed");
            return result;
        }

        // 2. 预处理
        preprocess(*frame->mat, static_cast<float*>(d_input_));
        
        cudaStreamSynchronize(stream_);

        // 3. 设置输入输出地址（TensorRT 10: enqueueV3）
        context_->setTensorAddress(input_name_.c_str(), d_input_);
        context_->setTensorAddress(boxes_name_.c_str(), d_boxes_);
        context_->setTensorAddress(scores_name_.c_str(), d_scores_);
        context_->setTensorAddress(classes_name_.c_str(), d_classes_);

        // 4. 执行推理
        auto t0 = std::chrono::high_resolution_clock::now();
        
        if (!context_->enqueueV3(stream_)) {
            LOG_ERROR_FMT("[DetectionInfer] enqueueV3 failed");
            return result;
        }
        cudaStreamSynchronize(stream_);

        // 在 cudaStreamSynchronize 之后，postprocess 之前添加：

        // 打印前 10 个原始值
        std::cout << "[DEBUG] Raw h_scores[0:10]: ";
        for (int i = 0; i < std::min(10, MAX_DETS); ++i) {
            std::cout << h_scores_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "[DEBUG] Raw h_classes[0:10]: ";
        for (int i = 0; i < std::min(10, MAX_DETS); ++i) {
            std::cout << h_classes_[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "[DEBUG] Raw h_boxes[0:4]: ";
        for (int i = 0; i < std::min(4, MAX_DETS * 4); ++i) {
            std::cout << h_boxes_[i] << " ";
        }
        std::cout << std::endl;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        float infer_ms = std::chrono::duration<float, std::milli>(t1 - t0).count();

        // 5. 获取输出形状（TensorRT 10 API）
        nvinfer1::Dims out_boxes_dims = context_->getTensorShape(boxes_name_.c_str());
        nvinfer1::Dims out_scores_dims = context_->getTensorShape(scores_name_.c_str());
        nvinfer1::Dims out_classes_dims = context_->getTensorShape(classes_name_.c_str());
        LOG_DEBUG_FMT("[DetectionInfer] Output shapes: boxes=[{}] scores=[{}] classes=[{}]", 
                      out_boxes_dims.d[0], out_scores_dims.d[0], out_classes_dims.d[0]);

        // 7. 拷贝输出回 Host（先拷贝全部 MAX_DETS，再推断实际数量）
        cudaMemcpyAsync(h_boxes_.data(), d_boxes_, out_boxes_size_,
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_scores_.data(), d_scores_, out_scores_size_,
                        cudaMemcpyDeviceToHost, stream_);
        cudaMemcpyAsync(h_classes_.data(), d_classes_, out_classes_size_,
                        cudaMemcpyDeviceToHost, stream_);
        cudaStreamSynchronize(stream_);

        // 8. 推断实际检测数（找到第一个 score < 0.001 的位置）
        int num_dets = 0;
        for (int i = 0; i < MAX_DETS; ++i) {
            if (h_scores_[i] < 0.001f || std::isnan(h_scores_[i])) break;
            num_dets++;
        }

        LOG_DEBUG_FMT("[DetectionInfer] Valid detections: {}", num_dets);

        // 调试打印
        std::cout << "[DEBUG] Raw h_scores[0:20]: ";
        for (int i = 0; i < std::min(20, MAX_DETS); ++i) {
            std::cout << h_scores_[i] << " ";
        }
        std::cout << std::endl;

        if (num_dets <= 0) {
            LOG_DEBUG_FMT("[DetectionInfer] No detections");
            return result;
        }

        // 9. 调试打印第一个检测
        LOG_DEBUG_FMT("[DetectionInfer] First: cls={} conf={:.3f} box=[{:.1f},{:.1f},{:.1f},{:.1f}]",
                      h_classes_[0], h_scores_[0],
                      h_boxes_[0], h_boxes_[1], h_boxes_[2], h_boxes_[3]);

        // 10. 缩放比例（使用原始帧尺寸，不是 640x640）
        // 注意：如果 resize_normalize 做了 letterbox，这里需要调整
        float scale_x = static_cast<float>(frame->source_mat->cols) / INPUT_W;
        float scale_y = static_cast<float>(frame->source_mat->rows) / INPUT_H;
        LOG_DEBUG_FMT("[DetectionInfer] Scale: {}x{}", scale_x, scale_y);
        
        // 11. 后处理
        result->detections = postprocess(num_dets, scale_x, scale_y, 0.25f);
        
        LOG_INFO_FMT("[DetectionInfer] ts={}ms: {} dets, infer={:.2f}ms",
                     frame->timestamp_ms, result->detections.size(), infer_ms);
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] Inference exception: {}", e.what());
    }

    return result;
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

        // 5. 保存资源
        runtime_.reset(runtime);
        engine_.reset(engine);
        context_.reset(context);

        // 6. 创建 CUDA Stream
        cudaStreamCreate(&stream_);

        // 7. 打印 Tensor 信息（TensorRT 10 API）
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

        // 8. 计算缓冲区大小
        input_size_ = static_cast<size_t>(batch_size_) * 3 * INPUT_H * INPUT_W * sizeof(float);
        out_boxes_size_ = static_cast<size_t>(MAX_DETS) * 4 * sizeof(float);
        out_scores_size_ = static_cast<size_t>(MAX_DETS) * sizeof(float);
        out_classes_size_ = static_cast<size_t>(MAX_DETS) * sizeof(int64_t);

        // 9. 分配 GPU 内存
        cudaMalloc(&d_input_, input_size_);
        cudaMalloc(&d_boxes_, out_boxes_size_);
        cudaMalloc(&d_scores_, out_scores_size_);
        cudaMalloc(&d_classes_, out_classes_size_);

        // 10. 预分配 CPU 缓冲区
        h_boxes_.resize(MAX_DETS * 4);
        h_scores_.resize(MAX_DETS);
        h_classes_.resize(MAX_DETS);

        LOG_INFO_FMT("[DetectionInfer] Engine loaded: {} (batch={}, max_dets={})",
                     engine_path, batch_size_, MAX_DETS);
        return true;

    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[DetectionInfer] initEngine exception: {}", e.what());
        return false;
    }
}

REGISTER_NODE("detection_infer", DetectionInferNode)

} // namespace nodes
} // namespace ai_stream