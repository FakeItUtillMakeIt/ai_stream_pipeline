// include/ai_stream/hal/detection_inference_engine_factory.h
// 检测推理引擎工厂——自动选择可用后端
#pragma once

#include "ai_stream/hal/i_detection_inference_engine.h"
#include <functional>
#include <unordered_map>

namespace ai_stream {
namespace hal {

/**
 * @brief 检测推理引擎后端类型
 */
enum class DetectionBackend {
    AUTO = 0,
    TENSORRT = 1,
    RKNN = 2,
    ASCEND = 3,
    CPU = 4
};

/**
 * @brief 检测推理引擎工厂
 *
 * 自动检测可用后端并按优先级选择：
 * TensorRT > RKNN > Ascend > CPU
 */
class DetectionInferenceEngineFactory {
public:
    static DetectionInferenceEngineFactory& instance();

    /**
     * @brief 注册后端
     * @param type 后端类型
     * @param creator 创建函数
     */
    void registerBackend(DetectionBackend type, std::function<DetectionInferenceEnginePtr()> creator);

    /**
     * @brief 创建推理引擎
     * @param type 后端类型（AUTO 表示自动选择）
     * @return 推理引擎实例，失败返回 nullptr
     */
    DetectionInferenceEnginePtr create(DetectionBackend type = DetectionBackend::AUTO) const;

    /**
     * @brief 获取可用后端列表
     */
    std::vector<std::pair<DetectionBackend, std::string>> getAvailableBackends() const;

    /**
     * @brief 检查指定后端是否可用
     */
    bool isBackendAvailable(DetectionBackend type) const;

private:
    DetectionInferenceEngineFactory() = default;
    std::unordered_map<DetectionBackend, std::function<DetectionInferenceEnginePtr()>> creators_;
};

/**
 * @brief 注册检测推理后端宏
 */
#define REGISTER_DETECTION_INFERENCE_BACKEND(type, ClassName) \
    static bool _detection_backend_registered_##ClassName = [](){ \
        DetectionInferenceEngineFactory::instance().registerBackend(type, [](){ \
            return std::make_shared<ClassName>(); \
        }); \
        return true; \
    }();

} // namespace hal
} // namespace ai_stream
