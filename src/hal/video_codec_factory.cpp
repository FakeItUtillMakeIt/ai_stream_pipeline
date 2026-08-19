// src/hal/video_codec_factory.cpp
// 视频编解码工厂实现
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

VideoCodecFactory& VideoCodecFactory::instance() {
    static VideoCodecFactory inst;
    return inst;
}

void VideoCodecFactory::registerBackend(VideoCodecBackend type, Creator creator) {
    creators_[type] = std::move(creator);
    LOG_DEBUG_FMT("[VideoCodecFactory] Registered backend: {}", static_cast<int>(type));
}

VideoCodecPtr VideoCodecFactory::create(VideoCodecBackend type) {
    if (type == VideoCodecBackend::AUTO) {
        // 优先级：NVDEC > MPP > DVPP > FFmpeg
        std::vector<VideoCodecBackend> priority = {
            VideoCodecBackend::NVDEC,
            VideoCodecBackend::MPP,
            VideoCodecBackend::DVPP,
            VideoCodecBackend::FFMPEG
        };
        for (auto backend : priority) {
            auto it = creators_.find(backend);
            if (it != creators_.end()) {
                auto codec = it->second();
                if (codec && codec->isAvailable()) {
                    LOG_INFO_FMT("[VideoCodecFactory] Auto-selected backend: {}", codec->getName());
                    return codec;
                }
            }
        }
        LOG_ERROR("[VideoCodecFactory] No video codec backend available");
        return nullptr;
    }

    auto it = creators_.find(type);
    if (it == creators_.end()) {
        LOG_ERROR_FMT("[VideoCodecFactory] Backend not registered: {}", static_cast<int>(type));
        return nullptr;
    }

    auto codec = it->second();
    if (!codec || !codec->isAvailable()) {
        LOG_ERROR_FMT("[VideoCodecFactory] Backend not available: {}", static_cast<int>(type));
        return nullptr;
    }

    return codec;
}

std::vector<std::pair<VideoCodecBackend, std::string>> VideoCodecFactory::getAvailableBackends() const {
    std::vector<std::pair<VideoCodecBackend, std::string>> result;
    for (const auto& [type, creator] : creators_) {
        auto codec = creator();
        if (codec && codec->isAvailable()) {
            result.emplace_back(type, codec->getName());
        }
    }
    return result;
}

bool VideoCodecFactory::isBackendAvailable(VideoCodecBackend type) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) return false;
    auto codec = it->second();
    return codec && codec->isAvailable();
}

} // namespace hal
} // namespace ai_stream
