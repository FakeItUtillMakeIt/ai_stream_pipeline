// src/hal/tensorrt/trt_core.h
// TensorRT 公共内核（内部实现，不进公开 API）——
// 封装各后端重复的引擎生命周期、设备缓冲区与推理执行机械，
// 供本目录下各后端引擎组合使用（composition over inheritance）。
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

/** Tensor 信息（避免在头文件泄漏 NvInfer 类型） */
struct TrtTensorMeta {
    std::string name;
    bool is_input = false;
    int nb_dims = 0;
    int64_t d[8] = {0};

    /** 元素数；动态维度(-1)按 1 计 */
    size_t elementsIgnoringDynamic() const {
        size_t n = 1;
        for (int i = 0; i < nb_dims && i < 8; ++i) {
            n *= (d[i] > 0) ? static_cast<size_t>(d[i]) : 1UL;
        }
        return n;
    }
};

/**
 * @brief TensorRT 引擎公共内核（RAII）
 *
 * 职责：
 * - .engine 文件反序列化 → runtime/engine/context 三件套生命周期
 * - 按名称的设备缓冲区分配注册表（析构统一释放）
 * - 动态输入 shape 设置与地址绑定
 * - enqueueV3 执行与流同步
 *
 * 非职责（留给具体后端）：领域预处理/后处理、结果解码。
 */
class TrtCore {
public:
    TrtCore();
    ~TrtCore();

    TrtCore(const TrtCore&) = delete;
    TrtCore& operator=(const TrtCore&) = delete;

    /** 从 .engine 文件加载；log_tag 用于日志前缀。成功后可执行推理。 */
    bool loadEngine(const std::string& engine_path, const char* log_tag);

    bool isLoaded() const;

    /** 枚举全部 IO tensor 元信息 */
    std::vector<TrtTensorMeta> tensors() const;

    /**
     * @brief 设置动态输入 shape
     * @param dims 维度数组（长度 nb_dims）
     * @return 是否成功
     */
    bool setInputShape(const std::string& name, const int64_t* dims, int nb_dims);

    /** 绑定 tensor 地址 */
    bool setAddress(const std::string& name, void* device_ptr);

    /**
     * @brief 分配（或复用足够大的）设备缓冲区并绑定到同名 tensor
     * @return 设备指针，失败返回 nullptr
     */
    void* allocBuffer(const std::string& tensor_name, size_t bytes);

    /** 已分配缓冲区指针（未分配返回 nullptr） */
    void* buffer(const std::string& tensor_name) const;

    /** 已分配缓冲区大小（未分配返回 0） */
    size_t bufferSize(const std::string& tensor_name) const;

    /** 在已设置的流上执行推理 */
    bool enqueue();

    /** 设置后续 enqueue 使用的流（void* 即 cudaStream_t） */
    void setStream(void* stream);
    void* stream() const;

    /** 同步指定流（nullptr 时同步内部流） */
    bool synchronize();

    // 原生句柄（CUDA Graph 等高级用途）
    void* context() const;   // nvinfer1::IExecutionContext*
    void* engine() const;    // nvinfer1::ICudaEngine*

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace hal
} // namespace ai_stream
