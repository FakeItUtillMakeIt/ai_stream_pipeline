// src/nodes/infer/detection_infer.h
#pragma once

#include "ai_stream/nodes/i_infer_node.h"
#include "src/core/frame_queue.h"
#include <thread>
#include <atomic>
#include <vector>
#include <string>
#include <memory>

// 前向声明 TensorRT 类型
namespace nvinfer1 {
    class ICudaEngine;
    class IExecutionContext;
}

namespace ai_stream {
namespace nodes {

/**
 * @brief 检测器推理节点
 *
 * 专门用于目标检测任务的推理节点，实现IInferNode接口。
 * 支持TensorRT引擎加载和目标检测推理。
 */
class DetectionInferNode : public IInferNode {
public:
    DetectionInferNode();
    ~DetectionInferNode() override;

    // IInferNode 接口实现
    bool loadModel(const std::string& model_path) override;
    void setPrecision(const std::string& precision) override;
    void setBatchSize(int batch_size) override;
    std::pair<int, int> getInputSize() const override;
    std::vector<std::string> getClassNames() const override;

    void setDetectorType(DetectorType type) override { detector_type_ = type; }
    DetectorType getDetectorType() const override { return detector_type_; }

    // Node 接口实现
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    /**
     * @brief 推理工作线程函数
     *
     * 从队列中获取帧数据，执行推理，广播结果
     */
    void inferLoop();

    /**
     * @brief 初始化TensorRT引擎
     * @param engine_path 引擎文件路径
     * @return 初始化是否成功
     */
    bool initEngine(const std::string& engine_path);

    /**
     * @brief 执行单个帧的推理
     * @param frame 视频帧数据
     * @return 推理结果包
     */
    std::shared_ptr<core::InferenceResultPacket> processFrame(
        std::shared_ptr<core::VideoFramePacket> frame);

    // TensorRT 相关成员
    std::unique_ptr<nvinfer1::ICudaEngine> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> context_;

    // 推理配置
    int input_width_ = 640;
    int input_height_ = 640;
    int batch_size_ = 1;
    std::string precision_ = "fp16";
    DetectorType detector_type_ = DetectorType::DETECTION;

    // 类别名称
    std::vector<std::string> class_names_ = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
        "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
        "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
        "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
        "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
        "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
        "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
        "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier",
        "toothbrush"
    };

    // 线程和数据队列
    core::BoundedQueue<std::shared_ptr<core::VideoFramePacket>> queue_{5};
    std::thread worker_;
    std::atomic<bool> running_{false};
};

} // namespace nodes
} // namespace ai_stream