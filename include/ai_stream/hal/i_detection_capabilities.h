// include/ai_stream/hal/i_detection_capabilities.h
// 检测推理后端的可选能力接口（避免把厂商专有细节泄漏进通用检测接口）
#pragma once

#include <cstddef>
#include <cstdint>

namespace ai_stream {
namespace hal {

/**
 * @brief CUDA Graph / 设备端 workspace 能力（TensorRT 专有优化）
 *
 * 需要 CUDA Graph 捕获、预分配并绑定执行期 workspace 的后端实现此接口。
 * 不支持的后端无需实现，调用方通过 dynamic_cast 探测。
 */
class IGraphCapturable {
public:
    virtual ~IGraphCapturable() = default;

    // 原始执行上下文指针（如 nvinfer1::IExecutionContext*），供 CUDA Graph 捕获使用
    virtual void* getRawContext() const = 0;
    // 原始引擎指针
    virtual void* getRawEngine() const = 0;
    // 执行推理所需的设备端 workspace 大小
    virtual size_t getDeviceMemorySize() const = 0;
    // 按当前 input shape 重新计算所需设备内存（含 activation + scratch）
    virtual size_t updateDeviceMemorySizeForShapes() = 0;
    // 设置预分配的设备端 workspace
    virtual bool setDeviceMemory(void* ptr) = 0;
    // 设置预分配的设备端 workspace 及其大小（TRT 10.3+ 推荐）
    virtual bool setDeviceMemoryV2(void* ptr, int64_t size) = 0;
};

/**
 * @brief NV12 直通输入能力（部分硬件后端，如地平线 BPU）
 */
class INv12Input {
public:
    virtual ~INv12Input() = default;

    /**
     * @brief 设置紧凑 NV12 输入（Y(w*h) 紧接 UV(w*h/2)）
     * @return 后端是否接受该输入；不接受时调用方回退常规预处理
     */
    virtual bool setNv12Input(const uint8_t* nv12, int width, int height) = 0;
};

} // namespace hal
} // namespace ai_stream
