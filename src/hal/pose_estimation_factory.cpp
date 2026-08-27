// src/hal/pose_estimation_factory.cpp
// 姿态估计引擎工厂实现
#include "ai_stream/hal/pose_estimation_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

PoseEstimationFactory& PoseEstimationFactory::instance() {
    static PoseEstimationFactory inst;
    return inst;
}

void PoseEstimationFactory::registerBackend(PoseEstimationBackend type, Creator creator) {
    creators_[type] = std::move(creator);
    LOG_DEBUG_FMT("[PoseFactory] Registered backend: {}", static_cast<int>(type));
}

PoseEstimationEnginePtr PoseEstimationFactory::create(PoseEstimationBackend type) {
    if (type == PoseEstimationBackend::AUTO) {
        // 优先级：TensorRT > RKNN > Ascend > CPU
        std::vector<PoseEstimationBackend> priority = {
            PoseEstimationBackend::TENSORRT,
            PoseEstimationBackend::RKNN,
            PoseEstimationBackend::ASCEND,
            PoseEstimationBackend::CPU
        };
        for (auto backend : priority) {
            auto it = creators_.find(backend);
            if (it != creators_.end()) {
                auto engine = it->second();
                if (engine && engine->isAvailable()) {
                    LOG_INFO_FMT("[PoseFactory] Auto-selected backend: {}", engine->getBackendName());
                    return engine;
                }
            }
        }
        LOG_ERROR("[PoseFactory] No pose estimation backend available");
        return nullptr;
    }

    auto it = creators_.find(type);
    if (it == creators_.end()) {
        LOG_ERROR_FMT("[PoseFactory] Backend not registered: {}", static_cast<int>(type));
        return nullptr;
    }

    auto engine = it->second();
    if (!engine || !engine->isAvailable()) {
        LOG_ERROR_FMT("[PoseFactory] Backend not available: {}", static_cast<int>(type));
        return nullptr;
    }

    return engine;
}

std::vector<std::pair<PoseEstimationBackend, std::string>> PoseEstimationFactory::getAvailableBackends() const {
    std::vector<std::pair<PoseEstimationBackend, std::string>> result;
    for (const auto& [type, creator] : creators_) {
        auto engine = creator();
        if (engine && engine->isAvailable()) {
            result.emplace_back(type, engine->getBackendName());
        }
    }
    return result;
}

bool PoseEstimationFactory::isBackendAvailable(PoseEstimationBackend type) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) return false;
    auto engine = it->second();
    return engine && engine->isAvailable();
}

} // namespace hal
} // namespace ai_stream
