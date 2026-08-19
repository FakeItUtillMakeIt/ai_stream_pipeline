// src/nodes/infer/detection_infer.h
// 检测推理节点——使用 HAL 抽象接口，支持多后端切换
// 【加速优化】Pinned Memory + CUDA Graph + 双流异步
#pragma once

#include "ai_stream/nodes/i_infer_node.h"
#include "ai_stream/core/bounded_queue.h"
#include "ai_stream/hal/i_detection_inference_engine.h"
#include "ai_stream/hal/detection_inference_engine_factory.h"

#include <cuda_runtime_api.h>

#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <deque>

namespace ai_stream {
namespace nodes {

class DetectionInferNode : public IInferNode {
public:
    DetectionInferNode();
    ~DetectionInferNode() override;

    bool loadModel(const std::string& model_path) override;
    void setPrecision(const std::string& precision) override;
    void setBatchSize(int batch_size) override;
    void setInputSize(int width, int height) override { input_width_ = width; input_height_ = height; }
    std::pair<int, int> getInputSize() const override;
    void setClassNames(const std::vector<std::string>& names) override { class_names_ = names; }
    std::vector<std::string> getClassNames() const override;

    void setDetectorType(DetectorType type) override { detector_type_ = type; }
    DetectorType getDetectorType() const override { return detector_type_; }

    // INT8 量化支持
    void setCalibrationData(const std::vector<std::string>& data) { calibration_data_ = data; }
    void setCalibrationCache(const std::string& path) { calibration_cache_path_ = path; }

    // CUDA Graph 开关
    void setCudaGraphEnabled(bool enable) { cuda_graph_enabled_ = enable; }

    // 设置推理后端类型
    void setInferenceBackend(hal::DetectionBackend backend) { backend_type_ = backend; }

    bool start() override;
    void stop() override;
    bool isRunning() const override { return running_.load(); }
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void inferLoop();
    bool initEngine(const std::string& engine_path);

    std::vector<std::shared_ptr<core::InferenceResultPacket>> processBatch(
        const std::vector<std::shared_ptr<core::VideoFramePacket>>& frames);

    // 双路径预处理
    void preprocessBatchCpu(const std::vector<cv::Mat*>& images, float* gpu_buffer, int batch_size);
    void preprocessBatchGpu(const std::vector<void*>& d_ptrs, const std::vector<size_t>& pitches,
                            float* gpu_buffer, int batch_size);

    std::vector<std::vector<core::InferenceResultPacket::BBox>> postprocessBatch(
        int batch_size, int total_dets,
        const float scale_x[], const float scale_y[], float conf_thresh,
        const float letter_scale[] = nullptr,
        const int letter_pad_x[] = nullptr,
        const int letter_pad_y[] = nullptr,
        const int letterbox_used[] = nullptr);

    // CUDA Graph 相关（TensorRT 特有优化）
    bool captureCudaGraph(int batch_size);
    bool executeCudaGraph();
    void destroyCudaGraph();

    // Pinned Memory 管理
    bool allocatePinnedMemory();
    void freePinnedMemory();

    // HAL 推理引擎
    hal::DetectionInferenceEnginePtr engine_;
    hal::DetectionBackend backend_type_ = hal::DetectionBackend::AUTO;

    // 双流架构：compute_stream 用于推理，transfer_stream 用于内存传输
    cudaStream_t compute_stream_ = nullptr;
    cudaStream_t transfer_stream_ = nullptr;

    // Tensor 名称
    std::string input_name_ = "images";
    std::string boxes_name_ = "det_boxes";
    std::string scores_name_ = "det_scores";
    std::string classes_name_ = "det_classes";
    std::string batch_ids_name_ = "det_batch_ids";
    std::string num_dets_name_ = "det_num_dets";

    // GPU 缓冲区
    void* d_input_ = nullptr;
    void* d_boxes_ = nullptr;
    void* d_scores_ = nullptr;
    void* d_classes_ = nullptr;
    void* d_batch_ids_ = nullptr;
    void* d_num_dets_ = nullptr;

    // 用于 GPU 路径的临时缓冲区
    void* d_preprocess_tmp_ = nullptr;
    size_t d_preprocess_tmp_size_ = 0;

    // Pinned Host Memory
    float* h_pinned_input_ = nullptr;
    float* h_pinned_boxes_ = nullptr;
    float* h_pinned_scores_ = nullptr;
    int64_t* h_pinned_classes_ = nullptr;
    int64_t* h_pinned_batch_ids_ = nullptr;
    int64_t* h_pinned_num_dets_ = nullptr;

    // CPU fallback 缓冲区
    std::vector<float> h_boxes_;
    std::vector<float> h_scores_;
    std::vector<int64_t> h_classes_;
    std::vector<int64_t> h_batch_ids_;
    int64_t h_num_dets_ = 0;

    // CUDA Graph 相关
    cudaGraph_t cuda_graph_ = nullptr;
    cudaGraphExec_t cuda_graph_exec_ = nullptr;
    int cuda_graph_batch_size_ = 0;
    std::atomic<bool> cuda_graph_enabled_{false};
    bool cuda_graph_ready_ = false;

    // 常量
    static constexpr int MAX_DETS = 200;

    size_t input_size_ = 0;
    size_t out_boxes_size_ = 0;
    size_t out_scores_size_ = 0;
    size_t out_classes_size_ = 0;
    size_t out_batch_ids_size_ = 0;
    size_t out_num_dets_size_ = 0;

    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 1;
    std::string precision_ = "fp16";
    DetectorType detector_type_ = DetectorType::DETECTION;

    // INT8 校准数据
    std::vector<std::string> calibration_data_;
    std::string calibration_cache_path_;

    std::vector<std::string> class_names_ = {
        "person", "head", "helmet", "clothes_red", "clothes_gray",
        "clothes_yellow", "clothes_blue", "clothes_similar",
        "clothes_reflective", "phone", "smoking", "fall_down",
        "safety_belt", "sleeping", "toy", "pad", "camera", "ring_light"
    };

    // 数据队列和线程
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> queue_{64};
    std::thread worker_;
    std::atomic<bool> running_{false};

    std::chrono::milliseconds batch_timeout_ms_{20};
    std::atomic<int> max_batch_size_{1};
};

} // namespace nodes
} // namespace ai_stream
