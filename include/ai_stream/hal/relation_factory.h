// include/ai_stream/hal/relation_factory.h
// 视觉关系后端工厂——根据编译选项和运行时配置创建具体后端
#pragma once

#include "ai_stream/hal/i_relation.h"
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace ai_stream {
namespace hal {

/**
 * @brief 关系后端类型
 */
enum class RelationBackend {
    AUTO,        // 自动选择可用后端
    TENSORRT,    // NVIDIA TensorRT
    ONNXRUNTIME, // 跨平台兜底
    CPU          // CPU 参考实现
};

/**
 * @brief 关系后端工厂
 */
class RelationFactory {
public:
    using Creator = std::function<RelationEnginePtr()>;

    static RelationFactory& instance();

    void registerBackend(RelationBackend type, Creator creator);
    RelationEnginePtr create(RelationBackend type = RelationBackend::AUTO);
    std::vector<std::pair<RelationBackend, std::string>> getAvailableBackends() const;
    bool isBackendAvailable(RelationBackend type) const;

private:
    RelationFactory() = default;
    std::unordered_map<RelationBackend, Creator> creators_;
};

// token 用 class_name 而不是 backend_type：枚举值带 "::"（RelationBackend::TENSORRT）
// 拼不进标识符，这与既有 action_recognition 工厂的宏同一写法。
#define REGISTER_RELATION_BACKEND(backend_type, class_name) \
    static struct _RelationRegistrar_##class_name { \
        _RelationRegistrar_##class_name() { \
            ai_stream::hal::RelationFactory::instance().registerBackend( \
                backend_type, []() -> ai_stream::hal::RelationEnginePtr { \
                    return std::make_unique<class_name>(); \
                }); \
        } \
    } _relation_registrar_##class_name;

} // namespace hal
} // namespace ai_stream
