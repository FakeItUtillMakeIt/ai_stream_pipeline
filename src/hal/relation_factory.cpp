// src/hal/relation_factory.cpp
// 视觉关系后端工厂实现
#include "ai_stream/hal/relation_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

RelationFactory& RelationFactory::instance() {
    static RelationFactory inst;
    return inst;
}

void RelationFactory::registerBackend(RelationBackend type, Creator creator) {
    creators_[type] = std::move(creator);
    LOG_DEBUG_FMT("[RelationFactory] Registered backend: {}", static_cast<int>(type));
}

RelationEnginePtr RelationFactory::create(RelationBackend type) {
    if (type == RelationBackend::AUTO) {
        std::vector<RelationBackend> priority = {
            RelationBackend::TENSORRT,
            RelationBackend::ONNXRUNTIME,
            RelationBackend::CPU
        };
        for (auto backend : priority) {
            auto it = creators_.find(backend);
            if (it == creators_.end())
                continue;
            auto engine = it->second();
            if (engine && engine->isAvailable()) {
                LOG_INFO_FMT("[RelationFactory] Auto-selected backend: {}",
                             engine->getBackendName());
                return engine;
            }
        }
        LOG_ERROR("[RelationFactory] No relation backend available");
        return nullptr;
    }

    auto it = creators_.find(type);
    if (it == creators_.end()) {
        LOG_ERROR_FMT("[RelationFactory] Backend not registered: {}", static_cast<int>(type));
        return nullptr;
    }
    auto engine = it->second();
    if (!engine || !engine->isAvailable()) {
        LOG_ERROR_FMT("[RelationFactory] Backend unavailable: {}", static_cast<int>(type));
        return nullptr;
    }
    return engine;
}

std::vector<std::pair<RelationBackend, std::string>> RelationFactory::getAvailableBackends() const {
    std::vector<std::pair<RelationBackend, std::string>> out;
    for (const auto& kv : creators_) {
        auto engine = kv.second();
        if (engine && engine->isAvailable())
            out.emplace_back(kv.first, engine->getBackendName());
    }
    return out;
}

bool RelationFactory::isBackendAvailable(RelationBackend type) const {
    auto it = creators_.find(type);
    if (it == creators_.end())
        return false;
    auto engine = it->second();
    return engine && engine->isAvailable();
}

} // namespace hal
} // namespace ai_stream
