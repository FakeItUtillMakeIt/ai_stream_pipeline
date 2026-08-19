// include/ai_stream/hal/inference_engine_factory.h
// 推理引擎工厂——根据编译选项和运行时配置创建具体后端
#pragma once

#include "ai_stream/hal/i_inference_engine.h"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace ai_stream {
namespace hal {

/**
 * @brief 推理引擎后端类型
 */
enum class InferenceBackend {
    AUTO,       // 自动选择可用后端
    TENSORRT,   // NVIDIA TensorRT
    RKNN,       // Rockchip RKNN (RK3588)
    ASCEND,     // Huawei Ascend CANN
    CPU         // CPU fallback（使用 OpenCV DNN 等）
};

/**
 * @brief 推理引擎工厂
 *
 * 注册和管理不同后端推理引擎的创建。
 * 使用静态工厂模式，后端通过 registerBackend 注册。
 */
class InferenceEngineFactory {
public:
    using Creator = std::function<InferenceEnginePtr()>;

    static InferenceEngineFactory& instance();

    /**
     * @brief 注册一个后端创建器
     */
    void registerBackend(InferenceBackend type, Creator creator);

    /**
     * @brief 创建指定类型的推理引擎
     */
    InferenceEnginePtr create(InferenceBackend type = InferenceBackend::AUTO);

    /**
     * @brief 获取所有可用后端名称
     */
    std::vector<std::pair<InferenceBackend, std::string>> getAvailableBackends() const;

    /**
     * @brief 检查指定后端是否可用
     */
    bool isBackendAvailable(InferenceBackend type) const;

private:
    InferenceEngineFactory() = default;
    std::unordered_map<InferenceBackend, Creator> creators_;
};

/**
 * @brief 注册助手宏——在 .cpp 文件中注册后端
 */
#define REGISTER_INFERENCE_BACKEND(backend_type, class_name) \
    static struct _InferenceBackendRegistrar_##class_name { \
        _InferenceBackendRegistrar_##class_name() { \
            InferenceEngineFactory::instance().registerBackend( \
                backend_type, []() -> InferenceEnginePtr { \
                    return std::make_unique<class_name>(); \
                }); \
        } \
    } _inference_backend_registrar_##class_name;

} // namespace hal
} // namespace ai_stream
