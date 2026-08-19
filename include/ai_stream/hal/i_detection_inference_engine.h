// include/ai_stream/hal/i_detection_inference_engine.h
// 检测推理引擎抽象接口——支持 GPU 内存直接操作和异步推理
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace ai_stream {
namespace hal {

/**
 * @brief 检测推理引擎配置
 */
struct DetectionInferenceConfig {
    std::string model_path;
    int input_width = 640;
    int input_height = 640;
    int max_batch_size = 1;
    int max_detections = 200;
    std::string precision = "fp16";
    int device_id = 0;
    bool enable_cuda_graph = false;
};

/**
 * @brief 检测推理引擎抽象接口
 *
 * 专为检测模型设计，支持：
 * - GPU 内存直接操作（零拷贝）
 * - 异步推理（CUDA stream）
 * - 多输出 tensor（boxes, scores, classes, batch_ids, num_dets）
 *
 * 所有检测推理后端（TensorRT、RKNN、Ascend）均实现此接口。
 */
class IDetectionInferenceEngine {
public:
    virtual ~IDetectionInferenceEngine() = default;

    /**
     * @brief 加载模型
     * @param config 推理配置
     * @return 是否加载成功
     */
    virtual bool loadModel(const DetectionInferenceConfig& config) = 0;

    /**
     * @brief 设置输入 tensor 地址（GPU 内存）
     * @param name tensor 名称
     * @param gpu_ptr GPU 内存指针
     * @return 是否设置成功
     */
    virtual bool setInputTensor(const std::string& name, void* gpu_ptr) = 0;

    /**
     * @brief 设置输出 tensor 地址（GPU 内存）
     * @param name tensor 名称
     * @param gpu_ptr GPU 内存指针
     * @return 是否设置成功
     */
    virtual bool setOutputTensor(const std::string& name, void* gpu_ptr) = 0;

    /**
     * @brief 获取输出 tensor 地址
     * @param name tensor 名称
     * @return GPU 内存指针（可能为空，表示由引擎管理）
     */
    virtual void* getOutputTensor(const std::string& name) = 0;

    /**
     * @brief 获取输出 tensor 大小
     * @param name tensor 名称
     * @return 字节数
     */
    virtual size_t getOutputTensorSize(const std::string& name) const = 0;

    /**
     * @brief 分配输出缓冲区（由引擎管理内存时使用）
     * @return 是否分配成功
     */
    virtual bool allocateOutputBuffers() = 0;

    /**
     * @brief 同步推理
     * @return 是否推理成功
     */
    virtual bool infer() = 0;

    /**
     * @brief 异步推理
     * @param stream CUDA stream
     * @return 是否推理成功
     */
    virtual bool inferAsync(void* stream) = 0;

    /**
     * @brief 同步等待推理完成
     * @param stream CUDA stream
     * @return 是否成功
     */
    virtual bool synchronize(void* stream) = 0;

    /**
     * @brief 获取输入 tensor 名称列表
     */
    virtual std::vector<std::string> getInputNames() const = 0;

    /**
     * @brief 获取输出 tensor 名称列表
     */
    virtual std::vector<std::string> getOutputNames() const = 0;

    /**
     * @brief 获取输入尺寸 [width, height]
     */
    virtual std::pair<int, int> getInputSize() const = 0;

    /**
     * @brief 获取最大批次大小
     */
    virtual int getMaxBatchSize() const = 0;

    /**
     * @brief 获取后端名称
     */
    virtual std::string getBackendName() const = 0;

    /**
     * @brief 检查后端是否可用
     */
    virtual bool isAvailable() const = 0;

    /**
     * @brief 获取原始上下文指针（用于高级优化，如 CUDA Graph）
     */
    virtual void* getRawContext() const = 0;

    /**
     * @brief 获取原始引擎指针（用于高级优化）
     */
    virtual void* getRawEngine() const = 0;
};

using DetectionInferenceEnginePtr = std::shared_ptr<IDetectionInferenceEngine>;

} // namespace hal
} // namespace ai_stream
