// src/nodes/postprocess/gpu_detection_post.h
#pragma once

#include "ai_stream/nodes/i_gpu_postprocess_node.h"

namespace ai_stream {
namespace nodes {

/**
 * @brief GPU 加速的检测后处理节点
 *
 * 使用 CUDA 实现高性能的 NMS、置信度过滤等后处理操作。
 * 适合大规模推理场景。
 */
class GPUDetectionPostProcessNode : public IGpuPostprocessNode {
public:
    GPUDetectionPostProcessNode();
    ~GPUDetectionPostProcessNode() override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    // IPostprocessNode 接口
    void setConfidenceThreshold(float threshold){conf_thresh_ = threshold;}
    float getConfidenceThreshold() const {return conf_thresh_;}
    void setNmsThreshold(float threshold){nms_thresh_ = threshold;}
    float getNmsThreshold() const {return nms_thresh_;}
    void setMaxDetections(int max_detections) {max_detections_ = max_detections;}
    void setClassWhitelist(const std::vector<std::string>& class_names) {class_whitelist_ = class_names;}
    void setTrackIdEnabled(bool enable) {track_id_enabled_ = enable;}
    void setPostProcessType(const std::string& type) {postprocess_type_ = type;}
    std::string getPostProcessType() const{return postprocess_type_;}

    // IGpuPostprocessNode 接口
    void setGpuDeviceId(int device_id) override;
    int getGpuDeviceId() const override;
    void setTensorRTPostprocessEnabled(bool enable) override;
    void setBatchSize(int batch_size) override;
    float getAverageLatencyMs() const override;

private:
    int device_id_;
    int batch_size_;
    cudaStream_t stream_;
    cudaEvent_t start_event_;
    cudaEvent_t stop_event_;

    // 性能统计
    std::atomic<int> total_processed_;
    std::atomic<int64_t> total_latency_ms_;
    std::atomic<bool> tensorrt_postprocess_enabled_;

    float conf_thresh_ = 0.5f;
    float nms_thresh_ = 0.4f;
    int max_detections_ = 100;
    std::vector<std::string> class_whitelist_;
    bool track_id_enabled_ = false;
    std::string postprocess_type_ = "detection";
};

} // namespace nodes
} // namespace ai_stream