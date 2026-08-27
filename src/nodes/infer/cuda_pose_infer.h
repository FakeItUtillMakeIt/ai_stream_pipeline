// src/nodes/infer/cuda_pose_infer.h
#pragma once

#include "ai_stream/nodes/i_infer_node.h"
#include "ai_stream/core/bounded_queue.h"
#include "ai_stream/hal/i_pose_estimation.h"
#include "ai_stream/hal/pose_estimation_factory.h"

#include <cuda_runtime_api.h>

#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <deque>

namespace ai_stream {
namespace nodes {

class CudaPoseInferNode : public IInferNode {
public:
    CudaPoseInferNode();
    ~CudaPoseInferNode() override;

    bool loadModel(const std::string& model_path) override;
    void setPrecision(const std::string& precision) override;
    void setBatchSize(int batch_size) override;
    void setInputSize(int width, int height) override { input_width_ = width; input_height_ = height;}
    std::pair<int, int> getInputSize() const override;
    void setClassNames(const std::vector<std::string>& names) override { class_names_ = names; }
    std::vector<std::string> getClassNames() const override;

    void setDetectorType(DetectorType type) override { detector_type_ = type; }
    DetectorType getDetectorType() const override { return detector_type_; }

    bool start() override;
    void stop() override;
    bool isRunning() const override{return running_.load();}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void inferLoop();

    // 处理单帧 InferenceResultPacket（GPU 预处理路径）
    void processFrame(std::shared_ptr<core::InferenceResultPacket> packet);

    // GPU 预处理：确保原图上传缓冲区足够
    void ensureSourceBuffer(int w, int h);

    // HAL 姿态估计引擎后端（GPU 预处理 kernel 由后端实现）
    hal::PoseEstimationEnginePtr engine_;

    // 原图上传缓冲区
    void* d_source_img_ = nullptr;   // 原图 GPU 缓冲区 (uint8 BGR)
    size_t source_buffer_size_ = 0;
    cudaStream_t stream_ = nullptr;

    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 8;   // 单帧最大人数
    std::string precision_ = "fp16";
    DetectorType detector_type_ = DetectorType::POSE;

    // 过滤参数
    float conf_thresh_ = 0.25f;      // person 置信度阈值
    float kpt_conf_thresh_ = 0.5f;   // 关键点可见度阈值
    int person_class_id_ = 0;        // person 的 class_id

    std::vector<std::string> class_names_ = {
        "person"
    };

    // 数据队列和线程
    core::BoundedQueue<std::shared_ptr<core::InferenceResultPacket>> queue_{64};
    std::thread worker_;
    std::atomic<bool> running_{false};

    // 耗时统计
    uint64_t in_time_ms_ = 0;
};

} // namespace nodes
} // namespace ai_stream
