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
    std::lock_guard<std::mutex> lock(cache_mutex_);
    availability_cache_.erase(type);
    LOG_INFO_FMT("[DetectionInferenceEngineFactory] Registered backend: {}", static_cast<int>(type));
}

DetectionInferenceEnginePtr DetectionInferenceEngineFactory::create(DetectionBackend type) const {
    if (type == DetectionBackend::AUTO) {
        // 按优先级自动选择：TensorRT > RKNN > Ascend > CPU
        static const DetectionBackend priority[] = {
            DetectionBackend::TENSORRT,
            DetectionBackend::RKNN,
            DetectionBackend::HORIZON,
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
        {DetectionBackend::HORIZON, "Horizon"},
        {DetectionBackend::ASCEND, "Ascend"},
        {DetectionBackend::CPU, "CPU"}
    };
    for (const auto& [type, creator] : creators_) {
        auto it = names.find(type);
        if (it != names.end() && isBackendAvailable(type)) {
            result.emplace_back(type, it->second);
        }
    }
    return result;
}

bool DetectionInferenceEngineFactory::isBackendAvailable(DetectionBackend type) const {
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        auto cit = availability_cache_.find(type);
        if (cit != availability_cache_.end()) {
            return cit->second;
        }
    }
    // 缓存未命中：在锁外探测（避免持锁 dlopen），再回写缓存
    auto it = creators_.find(type);
    bool avail = false;
    if (it != creators_.end()) {
        auto engine = it->second();
        avail = engine && engine->isAvailable();
    }
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        availability_cache_[type] = avail;
    }
    return avail;
}

} // namespace hal
} // namespace ai_stream
