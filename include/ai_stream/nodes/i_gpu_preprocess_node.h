// include/ai_stream/nodes/i_gpu_preprocess_node.h
#pragma once

#include "i_preprocess_node.h"

namespace ai_stream {
namespace nodes {

/**
 * @brief GPU 预处理节点接口
 *
 * 基于 CUDA 的预处理加速接口，用于高性能推理场景。
 * 继承自 IPreprocessNode，添加 GPU 相关配置。
 *
 * 注意：本头文件不包含 CUDA 头文件（公开头文件不得泄漏平台 SDK），
 * CUDA 流以 void* 传递，与 HAL 接口约定一致。
 */
class IGpuPreprocessNode : public IPreprocessNode {
public:
    using IPreprocessNode::IPreprocessNode;

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
     * @brief 设置是否使用异步处理
     * @param async 是否启用异步 CUDA 流
     */
    virtual void setAsyncProcessing(bool async) = 0;

    /**
     * @brief 设置 CUDA 流
     * @param stream 外部 CUDA 流（cudaStream_t 以 void* 传递）
     */
    virtual void setCudaStream(void* stream) = 0;

    /**
     * @brief 获取处理延迟统计
     * @return 平均处理延迟（毫秒）
     */
    virtual float getAverageLatencyMs() const = 0;

    /**
     * @brief 设置是否使用 TensorRT 预处理
     * @param enable 是否启用 TensorRT 预处理
     */
    virtual void setTensorRTPreprocessEnabled(bool enable) = 0;
};

} // namespace nodes
} // namespace ai_stream