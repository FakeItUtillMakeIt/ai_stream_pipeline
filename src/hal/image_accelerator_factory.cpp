// src/hal/image_accelerator_factory.cpp
// 图像加速器工厂实现
#include "ai_stream/hal/image_accelerator_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

ImageAcceleratorFactory& ImageAcceleratorFactory::instance() {
    static ImageAcceleratorFactory inst;
    return inst;
}

void ImageAcceleratorFactory::registerBackend(ImageAcceleratorBackend type, Creator creator) {
    creators_[type] = std::move(creator);
    LOG_DEBUG_FMT("[ImageAccelFactory] Registered backend: {}", static_cast<int>(type));
}

ImageAcceleratorPtr ImageAcceleratorFactory::create(ImageAcceleratorBackend type) {
    if (type == ImageAcceleratorBackend::AUTO) {
        // 优先级：NPP > RGA > DVPP > CPU
        std::vector<ImageAcceleratorBackend> priority = {
            ImageAcceleratorBackend::NPP,
            ImageAcceleratorBackend::RGA,
            ImageAcceleratorBackend::DVPP,
            ImageAcceleratorBackend::CPU
        };
        for (auto backend : priority) {
            auto it = creators_.find(backend);
            if (it != creators_.end()) {
                auto accel = it->second();
                if (accel && accel->isAvailable()) {
                    LOG_INFO_FMT("[ImageAccelFactory] Auto-selected backend: {}", accel->getName());
                    return accel;
                }
            }
        }
        LOG_ERROR("[ImageAccelFactory] No image accelerator backend available");
        return nullptr;
    }

    auto it = creators_.find(type);
    if (it == creators_.end()) {
        LOG_ERROR_FMT("[ImageAccelFactory] Backend not registered: {}", static_cast<int>(type));
        return nullptr;
    }

    auto accel = it->second();
    if (!accel || !accel->isAvailable()) {
        LOG_ERROR_FMT("[ImageAccelFactory] Backend not available: {}", static_cast<int>(type));
        return nullptr;
    }

    return accel;
}

std::vector<std::pair<ImageAcceleratorBackend, std::string>> ImageAcceleratorFactory::getAvailableBackends() const {
    std::vector<std::pair<ImageAcceleratorBackend, std::string>> result;
    for (const auto& [type, creator] : creators_) {
        auto accel = creator();
        if (accel && accel->isAvailable()) {
            result.emplace_back(type, accel->getName());
        }
    }
    return result;
}

bool ImageAcceleratorFactory::isBackendAvailable(ImageAcceleratorBackend type) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) return false;
    auto accel = it->second();
    return accel && accel->isAvailable();
}

} // namespace hal
} // namespace ai_stream
