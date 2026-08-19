// include/ai_stream/hal/action_recognition_factory.h
// 动作识别引擎工厂——根据编译选项和运行时配置创建具体后端
#pragma once

#include "ai_stream/hal/i_action_recognition.h"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace ai_stream {
namespace hal {

/**
 * @brief 动作识别后端类型
 */
enum class ActionRecognitionBackend {
    AUTO,       // 自动选择可用后端
    TENSORRT,   // NVIDIA TensorRT
    RKNN,       // Rockchip RKNN
    ASCEND,     // Huawei Ascend OM
    CPU         // CPU fallback
};

/**
 * @brief 动作识别引擎工厂
 */
class ActionRecognitionFactory {
public:
    using Creator = std::function<ActionRecognitionEnginePtr()>;

    static ActionRecognitionFactory& instance();

    void registerBackend(ActionRecognitionBackend type, Creator creator);
    ActionRecognitionEnginePtr create(ActionRecognitionBackend type = ActionRecognitionBackend::AUTO);
    std::vector<std::pair<ActionRecognitionBackend, std::string>> getAvailableBackends() const;
    bool isBackendAvailable(ActionRecognitionBackend type) const;

private:
    ActionRecognitionFactory() = default;
    std::unordered_map<ActionRecognitionBackend, Creator> creators_;
};

#define REGISTER_ACTION_RECOGNITION_BACKEND(backend_type, class_name) \
    static struct _ActionRecRegistrar_##class_name { \
        _ActionRecRegistrar_##class_name() { \
            ActionRecognitionFactory::instance().registerBackend( \
                backend_type, []() -> ActionRecognitionEnginePtr { \
                    return std::make_unique<class_name>(); \
                }); \
        } \
    } _action_rec_registrar_##class_name;

} // namespace hal
} // namespace ai_stream
