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
    std::lock_guard<std::mutex> lock(mutex_);
    creators_[type] = std::move(creator);
    availability_cache_.erase(type);
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
            Creator creator;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto it = creators_.find(backend);
                if (it != creators_.end()) {
                    creator = it->second;
                }
            }
            if (creator) {
                auto accel = creator();
                if (accel && accel->isAvailable()) {
                    LOG_INFO_FMT("[ImageAccelFactory] Auto-selected backend: {}", accel->getName());
                    return accel;
                }
            }
        }
        LOG_ERROR("[ImageAccelFactory] No image accelerator backend available");
        return nullptr;
    }

    Creator creator;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            creator = it->second;
        }
    }
    if (!creator) {
        LOG_ERROR_FMT("[ImageAccelFactory] Backend not registered: {}", static_cast<int>(type));
        return nullptr;
    }

    auto accel = creator();
    if (!accel || !accel->isAvailable()) {
        LOG_ERROR_FMT("[ImageAccelFactory] Backend not available: {}", static_cast<int>(type));
        return nullptr;
    }

    return accel;
}

std::vector<std::pair<ImageAcceleratorBackend, std::string>> ImageAcceleratorFactory::getAvailableBackends() const {
    std::vector<std::pair<ImageAcceleratorBackend, std::string>> result;
    std::lock_guard<std::mutex> lock(mutex_);
    for (const auto& [type, creator] : creators_) {
        auto cached = availability_cache_.find(type);
        if (cached == availability_cache_.end()) {
            auto accel = creator();
            const bool available = accel && accel->isAvailable();
            const std::string name = accel ? accel->getName() : std::string();
            cached = availability_cache_.emplace(type, std::make_pair(available, name)).first;
        }
        if (cached->second.first) {
            result.emplace_back(type, cached->second.second);
        }
    }
    return result;
}

bool ImageAcceleratorFactory::isBackendAvailable(ImageAcceleratorBackend type) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto cached = availability_cache_.find(type);
    if (cached != availability_cache_.end()) {
        return cached->second.first;
    }

    auto it = creators_.find(type);
    if (it == creators_.end()) {
        return false;
    }

    auto accel = it->second();
    const bool available = accel && accel->isAvailable();
    const std::string name = accel ? accel->getName() : std::string();
    availability_cache_.emplace(type, std::make_pair(available, name));
    return available;
}

} // namespace hal
} // namespace ai_stream
