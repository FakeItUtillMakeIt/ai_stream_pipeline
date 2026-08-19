// src/hal/action_recognition_factory.cpp
// 动作识别引擎工厂实现
#include "ai_stream/hal/action_recognition_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

ActionRecognitionFactory& ActionRecognitionFactory::instance() {
    static ActionRecognitionFactory inst;
    return inst;
}

void ActionRecognitionFactory::registerBackend(ActionRecognitionBackend type, Creator creator) {
    creators_[type] = std::move(creator);
    LOG_DEBUG_FMT("[ActionRecFactory] Registered backend: {}", static_cast<int>(type));
}

ActionRecognitionEnginePtr ActionRecognitionFactory::create(ActionRecognitionBackend type) {
    if (type == ActionRecognitionBackend::AUTO) {
        // 优先级：TensorRT > RKNN > Ascend > CPU
        std::vector<ActionRecognitionBackend> priority = {
            ActionRecognitionBackend::TENSORRT,
            ActionRecognitionBackend::RKNN,
            ActionRecognitionBackend::ASCEND,
            ActionRecognitionBackend::CPU
        };
        for (auto backend : priority) {
            auto it = creators_.find(backend);
            if (it != creators_.end()) {
                auto engine = it->second();
                if (engine && engine->isAvailable()) {
                    LOG_INFO_FMT("[ActionRecFactory] Auto-selected backend: {}", engine->getBackendName());
                    return engine;
                }
            }
        }
        LOG_ERROR("[ActionRecFactory] No action recognition backend available");
        return nullptr;
    }

    auto it = creators_.find(type);
    if (it == creators_.end()) {
        LOG_ERROR_FMT("[ActionRecFactory] Backend not registered: {}", static_cast<int>(type));
        return nullptr;
    }

    auto engine = it->second();
    if (!engine || !engine->isAvailable()) {
        LOG_ERROR_FMT("[ActionRecFactory] Backend not available: {}", static_cast<int>(type));
        return nullptr;
    }

    return engine;
}

std::vector<std::pair<ActionRecognitionBackend, std::string>> ActionRecognitionFactory::getAvailableBackends() const {
    std::vector<std::pair<ActionRecognitionBackend, std::string>> result;
    for (const auto& [type, creator] : creators_) {
        auto engine = creator();
        if (engine && engine->isAvailable()) {
            result.emplace_back(type, engine->getBackendName());
        }
    }
    return result;
}

bool ActionRecognitionFactory::isBackendAvailable(ActionRecognitionBackend type) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) return false;
    auto engine = it->second();
    return engine && engine->isAvailable();
}

} // namespace hal
} // namespace ai_stream
