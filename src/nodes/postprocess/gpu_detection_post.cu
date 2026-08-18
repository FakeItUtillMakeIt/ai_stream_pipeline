// src/nodes/postprocess/gpu_detection_post.cu
#include "gpu_detection_post.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "utils/cuda_check.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

namespace ai_stream {
namespace nodes {

namespace {

// CUDA 内核：置信度过滤
__global__ void confidenceFilterKernel(float* boxes, float* scores, int* class_ids,
                                       int max_boxes, float conf_threshold, int* suppress_flags) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_boxes) return;

    if (scores[idx] < conf_threshold) {
        suppress_flags[idx] = 1;  // 标记为需要抑制
    }
}

// CUDA 内核：IoU 计算
__device__ float calculateIoU(const float* box1, const float* box2) {
    float x1 = max(box1[0], box2[0]);
    float y1 = max(box1[1], box2[1]);
    float x2 = min(box1[0] + box1[2], box2[0] + box2[2]);
    float y2 = min(box1[1] + box1[3], box2[1] + box2[3]);

    float inter_area = max(0.0f, x2 - x1) * max(0.0f, y2 - y1);
    float box1_area = box1[2] * box1[3];
    float box2_area = box2[2] * box2[3];

    if (box1_area + box2_area - inter_area <= 0.0f) {
        return 0.0f;
    }
    return inter_area / (box1_area + box2_area - inter_area);
}

// CUDA 内核：NMS 实现
__global__ void nmsKernel(float* boxes, float* scores, int* class_ids,
                          int* keep_indices, int* keep_count,
                          int max_boxes, float nms_threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= max_boxes) return;

    // 简化的 NMS 实现，实际应用中需要更复杂的逻辑
    // 这里只是占位符
}

} // namespace

GPUDetectionPostProcessNode::GPUDetectionPostProcessNode()
    : core::QueuedNode<IGpuPostprocessNode>("GPUDetectionPostProcess"),
      device_id_(0),
      batch_size_(1),
      stream_(nullptr),
      total_processed_(0),
      total_latency_ms_(0.0f),
      tensorrt_postprocess_enabled_(false) {

    // 初始化 CUDA
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count > 0) {
        device_id_ = 0;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[GPUDetectionPostProcess] Node created with GPU device %d", device_id_);
    } else {
        LOG_WARN_FMT("[GPUDetectionPostProcess] No GPU device found, will fallback to CPU");
    }

    // 创建 CUDA 事件
    cudaEventCreate(&start_event_);
    cudaEventCreate(&stop_event_);
}

GPUDetectionPostProcessNode::~GPUDetectionPostProcessNode() {
    stop();

    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    cudaEventDestroy(start_event_);
    cudaEventDestroy(stop_event_);

    LOG_INFO_FMT("[GPUDetectionPostProcess] Node destroyed");
}

bool GPUDetectionPostProcessNode::onStartup() {
    if (!stream_) {
        CUDA_CHECK_BOOL(cudaStreamCreate(&stream_));
    }

    LOG_INFO_FMT("[GPUDetectionPostProcess] Node started");
    return true;
}

void GPUDetectionPostProcessNode::onShutdown() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
    }

    LOG_INFO_FMT("[GPUDetectionPostProcess] Node stopped");
}

void GPUDetectionPostProcessNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[GpuDetectionPostProcess] Received stream end");
        stop();
        broadcast(packet);
        return;
    }

    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }

    cudaEventRecord(start_event_, stream_);

    auto infer_result = std::static_pointer_cast<core::InferenceResultPacket>(packet);
    auto& dets = infer_result->detections;

    if (dets.empty()) {
        broadcast(packet);
        return;
    }

    // 1. 置信度过滤和类别白名单过滤
    dets.erase(std::remove_if(dets.begin(), dets.end(),
                              [this](const auto& box) {
                                  if (box.confidence < conf_thresh_) {
                                      return true;
                                  }
                                  if (!class_whitelist_.empty() &&
                                      std::find(class_whitelist_.begin(), class_whitelist_.end(), box.class_name) == class_whitelist_.end()) {
                                      return true;
                                  }
                                  return false;
                              }),
               dets.end());

    if (dets.empty()) {
        broadcast(packet);
        return;
    }

    // 2. 准备 GPU 数据
    size_t dets_size = dets.size() * sizeof(core::InferenceResultPacket::BBox);
    void *dets_gpu, *scores_gpu, *suppress_flags_gpu;
    int *keep_indices_gpu, keep_count_gpu = 0;

    CUDA_CHECK(cudaMalloc(&dets_gpu, dets_size));
    CUDA_CHECK(cudaMalloc(&scores_gpu, dets.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&suppress_flags_gpu, dets.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&keep_indices_gpu, dets.size() * sizeof(int)));

    // 3. 复制数据到 GPU（简化处理）
    // 实际应用中需要更复杂的内存布局
    CUDA_CHECK(cudaMemcpyAsync(dets_gpu, dets.data(), dets_size,
                               cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaMemsetAsync(suppress_flags_gpu, 0, dets.size() * sizeof(int), stream_));

    // 4. GPU 加速的置信度过滤
    dim3 block(256);
    dim3 grid((dets.size() + block.x - 1) / block.x);

    confidenceFilterKernel<<<grid, block, 0, stream_>>>(
        static_cast<float*>(dets_gpu), // boxes
        static_cast<float*>(scores_gpu), // scores
        nullptr, // class_ids
        dets.size(),
        conf_thresh_,
        static_cast<int*>(suppress_flags_gpu)
    );

    // 5. GPU 加速的 NMS
    // 简化处理，实际实现需要更复杂的算法
    nmsKernel<<<grid, block, 0, stream_>>>(
        static_cast<float*>(dets_gpu),
        static_cast<float*>(scores_gpu),
        nullptr, // class_ids
        keep_indices_gpu,
        &keep_count_gpu,
        dets.size(),
        nms_thresh_
    );

    // 6. 复制结果回主机
    CUDA_CHECK(cudaMemcpyAsync(dets.data(), dets_gpu, dets_size,
                               cudaMemcpyDeviceToHost, stream_));

    // 7. 应用 NMS 结果
    std::vector<core::InferenceResultPacket::BBox> nms_dets;
    for (int i = 0; i < keep_count_gpu && i < static_cast<int>(dets.size()) && i < max_detections_; ++i) {
        nms_dets.push_back(dets[i]);
    }
    dets = std::move(nms_dets);

    // 8. 清理 GPU 资源
    cudaFree(dets_gpu);
    cudaFree(scores_gpu);
    cudaFree(suppress_flags_gpu);
    cudaFree(keep_indices_gpu);

    cudaEventRecord(stop_event_, stream_);
    cudaEventSynchronize(stop_event_);

    float latency_ms;
    cudaEventElapsedTime(&latency_ms, start_event_, stop_event_);
    total_latency_ms_ += static_cast<int64_t>(latency_ms);
    total_processed_++;

    // 9. 跟踪ID处理（占位）
    if (track_id_enabled_) {
        LOG_INFO_FMT("[GPUDetectionPostProcess] Track ID enabled - need tracker integration");
    }

    broadcast(infer_result);
}

void GPUDetectionPostProcessNode::setGpuDeviceId(int device_id) {
    if (device_id_ != device_id) {
        device_id_ = device_id;
        cudaSetDevice(device_id_);
        LOG_INFO_FMT("[GPUDetectionPostProcess] Set GPU device ID: %d", device_id_);
    }
}

int GPUDetectionPostProcessNode::getGpuDeviceId() const {
    return device_id_;
}

void GPUDetectionPostProcessNode::setTensorRTPostprocessEnabled(bool enable) {
    tensorrt_postprocess_enabled_ = enable;
    LOG_INFO_FMT("[GPUDetectionPostProcess] Set TensorRT postprocess: %d", enable);
}

void GPUDetectionPostProcessNode::setBatchSize(int batch_size) {
    batch_size_ = batch_size;
    LOG_INFO_FMT("[GPUDetectionPostProcess] Set batch size: %d", batch_size);
}

float GPUDetectionPostProcessNode::getAverageLatencyMs() const {
    return total_processed_ > 0 ? total_latency_ms_ / total_processed_ : 0.0f;
}

// 注册节点
REGISTER_NODE("gpu_detection_post", GPUDetectionPostProcessNode)

} // namespace nodes
} // namespace ai_stream