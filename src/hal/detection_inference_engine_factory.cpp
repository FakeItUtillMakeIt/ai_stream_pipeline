// src/hal/detection_inference_engine_factory.cpp
// 检测推理引擎工厂实现
#include "ai_stream/hal/detection_inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

DetectionInferenceEngineFactory& DetectionInferenceEngineFactory::instance() {
    static DetectionInferenceEngineFactory factory;
    return factory;
}

void DetectionInferenceEngineFactory::registerBackend(
    DetectionBackend type, std::function<DetectionInferenceEnginePtr()> creator) {
    creators_[type] = std::move(creator);
    LOG_INFO_FMT("[DetectionInferenceEngineFactory] Registered backend: {}", static_cast<int>(type));
}

DetectionInferenceEnginePtr DetectionInferenceEngineFactory::create(DetectionBackend type) const {
    if (type == DetectionBackend::AUTO) {
        // 按优先级自动选择：TensorRT > RKNN > Ascend > CPU
        static const DetectionBackend priority[] = {
            DetectionBackend::TENSORRT,
            DetectionBackend::RKNN,
            DetectionBackend::ASCEND,
            DetectionBackend::CPU
        };
        for (auto backend : priority) {
            auto it = creators_.find(backend);
            if (it != creators_.end()) {
                auto engine = it->second();
                if (engine && engine->isAvailable()) {
                    LOG_INFO_FMT("[DetectionInferenceEngineFactory] Auto-selected backend: {}", static_cast<int>(backend));
                    return engine;
                }
            }
        }
        LOG_ERROR("[DetectionInferenceEngineFactory] No detection inference backend available");
        return nullptr;
    }

    auto it = creators_.find(type);
    if (it != creators_.end()) {
        return it->second();
    }

    LOG_ERROR_FMT("[DetectionInferenceEngineFactory] Backend not registered: {}", static_cast<int>(type));
    return nullptr;
}

std::vector<std::pair<DetectionBackend, std::string>> DetectionInferenceEngineFactory::getAvailableBackends() const {
    std::vector<std::pair<DetectionBackend, std::string>> result;
    static const std::unordered_map<DetectionBackend, std::string> names = {
        {DetectionBackend::TENSORRT, "TensorRT"},
        {DetectionBackend::RKNN, "RKNN"},
        {DetectionBackend::ASCEND, "Ascend"},
        {DetectionBackend::CPU, "CPU"}
    };
    for (const auto& [type, creator] : creators_) {
        auto it = names.find(type);
        if (it != names.end()) {
            auto engine = creator();
            if (engine && engine->isAvailable()) {
                result.emplace_back(type, it->second);
            }
        }
    }
    return result;
}

bool DetectionInferenceEngineFactory::isBackendAvailable(DetectionBackend type) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) return false;
    auto engine = it->second();
    return engine && engine->isAvailable();
}

} // namespace hal
} // namespace ai_stream
