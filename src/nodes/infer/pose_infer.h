// src/nodes/infer/pose_infer.h
#pragma once

#include "ai_stream/nodes/i_infer_node.h"
#include "ai_stream/core/bounded_queue.h"
#include "ai_stream/hal/i_pose_estimation.h"
#include "ai_stream/hal/pose_estimation_factory.h"

#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <deque>

namespace ai_stream {
namespace nodes {

class PoseInferNode : public IInferNode {
public:
    PoseInferNode();
    ~PoseInferNode() override;

    bool loadModel(const std::string& model_path) override;
    void setPrecision(const std::string& precision) override;
    void setBatchSize(int batch_size) override;
    void setDeviceId(int device_id) override { device_id_ = device_id; }
    void setInputSize(int width, int height) override { input_width_ = width; input_height_ = height; }
    std::pair<int, int> getInputSize() const override;
    void setClassNames(const std::vector<std::string>& names) override { class_names_ = names; }
    std::vector<std::string> getClassNames() const override;

    void setDetectorType(DetectorType type) override { detector_type_ = type; }
    DetectorType getDetectorType() const override { return detector_type_; }

    bool start() override;
    void stop() override;
    bool isRunning() const override{return running_.load();}
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    // 消费 InferenceResultPacket 时需要保住 source_frame 的设备端 BGR
    // （d_bgr_ptr），供 inferFromDeviceImage GPU 路径使用。
    bool acceptsGpuFrame() const override { return true; }

private:
    void inferLoop();

    // 处理单帧 InferenceResultPacket
    void processFrame(std::shared_ptr<core::InferenceResultPacket> packet);

    // GPU 路径：设备端 BGR 原图 + 检测框（boxes_7），后端内部完成
    // GPU crop/letterbox/normalize 与推理，随后解码关键点。
    // 不可用（无 d_bgr_ptr / 后端不支持）时返回 false，调用方回退 CPU 路径。
    bool inferFromGpuImage(std::shared_ptr<core::InferenceResultPacket> packet,
                           const std::vector<int>& person_indices,
                           int num_persons);

    // 从 source_mat 按检测框 crop + letterbox resize + normalize（host 输入路径）
    bool cropAndPreprocess(
        const cv::Mat& source_mat,
        const core::InferenceResultPacket::BBox& det,
        float* host_buffer,
        int slot_idx,
        float& out_scale,
        float& out_pad_x,
        float& out_pad_y);

    // HAL 姿态估计引擎后端（TensorRT/RKNN/Ascend 由工厂按编译配置选择）
    hal::PoseEstimationEnginePtr engine_;

    // 常量
    static constexpr int NUM_KEYPOINTS = 17;

    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 8;   // 单帧最大人数
    int device_id_ = 0;
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
