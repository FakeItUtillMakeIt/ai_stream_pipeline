// src/hal/inference_engine_factory.cpp
// 推理引擎工厂实现
#include "ai_stream/hal/inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

InferenceEngineFactory& InferenceEngineFactory::instance() {
    static InferenceEngineFactory factory;
    return factory;
}

void InferenceEngineFactory::registerBackend(InferenceBackend type, Creator creator) {
    creators_[type] = std::move(creator);
    LOG_INFO_FMT("[InferenceEngineFactory] Registered backend: {}", static_cast<int>(type));
}

InferenceEnginePtr InferenceEngineFactory::create(InferenceBackend type) {
    if (type == InferenceBackend::AUTO) {
        // 优先级：TensorRT > RKNN > Ascend > CPU
        std::vector<InferenceBackend> priority = {
            InferenceBackend::TENSORRT,
            InferenceBackend::RKNN,
            InferenceBackend::ASCEND,
            InferenceBackend::CPU
        };
        for (auto backend : priority) {
            auto it = creators_.find(backend);
            if (it != creators_.end()) {
                auto engine = it->second();
                if (engine && engine->isAvailable()) {
                    LOG_INFO_FMT("[InferenceEngineFactory] Auto-selected backend: {}", static_cast<int>(backend));
                    return engine;
                }
            }
        }
        LOG_ERROR("[InferenceEngineFactory] No inference backend available");
        return nullptr;
    }

    auto it = creators_.find(type);
    if (it != creators_.end()) {
        return it->second();
    }

    LOG_ERROR_FMT("[InferenceEngineFactory] Backend not registered: {}", static_cast<int>(type));
    return nullptr;
}

std::vector<std::pair<InferenceBackend, std::string>> InferenceEngineFactory::getAvailableBackends() const {
    std::vector<std::pair<InferenceBackend, std::string>> result;
    for (const auto& [type, creator] : creators_) {
        auto engine = creator();
        if (engine && engine->isAvailable()) {
            result.emplace_back(type, engine->getBackendName());
        }
    }
    return result;
}

bool InferenceEngineFactory::isBackendAvailable(InferenceBackend type) const {
    auto it = creators_.find(type);
    if (it != creators_.end()) {
        auto engine = it->second();
        return engine && engine->isAvailable();
    }
    return false;
}

} // namespace hal
} // namespace ai_stream
