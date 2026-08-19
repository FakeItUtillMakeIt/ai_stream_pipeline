// include/ai_stream/hal/i_inference_engine.h
// 推理引擎抽象接口——隔离 TensorRT / RKNN / Ascend 等后端
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <utility>

namespace ai_stream {
namespace hal {

/**
 * @brief 推理引擎配置
 */
struct InferenceConfig {
    std::string model_path;
    std::string input_name = "images";
    std::string output_name = "output0";
    int input_width = 640;
    int input_height = 640;
    int batch_size = 1;
    std::string precision = "fp16";  // fp32, fp16, int8
    int device_id = 0;
};

/**
 * @brief 推理引擎抽象接口
 *
 * 所有推理后端（TensorRT、RKNN、Ascend）均实现此接口，
 * 推理节点通过此接口调用推理，解除对具体后端的编译依赖。
 */
class IInferenceEngine {
public:
    virtual ~IInferenceEngine() = default;

    /**
     * @brief 加载模型
     * @param config 推理配置
     * @return 是否加载成功
     */
    virtual bool loadModel(const InferenceConfig& config) = 0;

    /**
     * @brief 执行推理
     * @param input_data 输入数据（主机内存，已预处理）
     * @param input_size 输入数据字节数
     * @param output_data 输出缓冲区（主机内存）
     * @param output_size 输出缓冲区字节数
     * @return 是否推理成功
     */
    virtual bool infer(const void* input_data, size_t input_size,
                       void* output_data, size_t output_size) = 0;

    /**
     * @brief 获取模型输入尺寸 [width, height]
     */
    virtual std::pair<int, int> getInputSize() const = 0;

    /**
     * @brief 获取批次大小
     */
    virtual int getBatchSize() const = 0;

    /**
     * @brief 获取后端名称（用于日志）
     */
    virtual std::string getBackendName() const = 0;

    /**
     * @brief 检查后端是否可用
     */
    virtual bool isAvailable() const = 0;
};

using InferenceEnginePtr = std::unique_ptr<IInferenceEngine>;

} // namespace hal
} // namespace ai_stream
