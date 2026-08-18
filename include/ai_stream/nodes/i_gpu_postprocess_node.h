// include/ai_stream/nodes/i_gpu_postprocess_node.h
#pragma once

#include "i_postprocess_node.h"

namespace ai_stream {
namespace nodes {

/**
 * @brief GPU 后处理节点接口
 *
 * 基于 CUDA 的后处理加速接口，用于高性能推理后处理。
 * 支持 GPU 加速的 NMS、置信度过滤等操作。
 */
class IGpuPostprocessNode : public IPostprocessNode {
public:
    using IPostprocessNode::IPostprocessNode;

    /**
     * @brief 设置 GPU 设备 ID
     * @param device_id CUDA 设备 ID
     */
    virtual void setGpuDeviceId(int device_id) = 0;

    /**
     * @brief 获取当前 GPU 设备 ID
     * @return CUDA 设备 ID
     */
    virtual int getGpuDeviceId() const = 0;

    /**
     * @brief 设置是否使用 TensorRT 后处理
     * @param enable 是否启用 TensorRT 后处理
     */
    virtual void setTensorRTPostprocessEnabled(bool enable) = 0;

    /**
     * @brief 设置后处理批处理大小
     * @param batch_size 批处理大小
     */
    virtual void setBatchSize(int batch_size) = 0;

    /**
     * @brief 获取处理延迟统计
     * @return 平均处理延迟（毫秒）
     */
    virtual float getAverageLatencyMs() const = 0;
};

} // namespace nodes
} // namespace ai_stream