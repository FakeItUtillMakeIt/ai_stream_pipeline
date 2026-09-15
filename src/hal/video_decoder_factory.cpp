// src/hal/video_decoder_factory.cpp
// 视频编解码工厂实现
#include "ai_stream/hal/video_decoder_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

VideoDecoderFactory& VideoDecoderFactory::instance() {
    static VideoDecoderFactory inst;
    return inst;
}

void VideoDecoderFactory::registerBackend(VideoDecoderBackend type, Creator creator) {
    creators_[type] = std::move(creator);
    LOG_DEBUG_FMT("[VideoDecoderFactory] Registered backend: {}", static_cast<int>(type));
}

VideoDecoderPtr VideoDecoderFactory::create(VideoDecoderBackend type) {
    if (type == VideoDecoderBackend::AUTO) {
        // 优先级：NVDEC(cuvid) > NVV4L2(GStreamer/Jetson) > V4L2(裸/Jetson) > MPP > DVPP > FFmpeg
        std::vector<VideoDecoderBackend> priority = {
            VideoDecoderBackend::NVDEC,
            VideoDecoderBackend::NVV4L2,
            VideoDecoderBackend::V4L2,
            VideoDecoderBackend::MPP,
            VideoDecoderBackend::HORIZON,
            VideoDecoderBackend::DVPP,
            VideoDecoderBackend::FFMPEG
        };
        for (auto backend : priority) {
            auto it = creators_.find(backend);
            if (it != creators_.end()) {
                auto codec = it->second();
                if (codec && codec->isAvailable()) {
                    LOG_INFO_FMT("[VideoDecoderFactory] Auto-selected backend: {}", codec->getName());
                    return codec;
                }
            }
        }
        LOG_ERROR("[VideoDecoderFactory] No video codec backend available");
        return nullptr;
    }

    auto it = creators_.find(type);
    if (it == creators_.end()) {
        LOG_ERROR_FMT("[VideoDecoderFactory] Backend not registered: {}", static_cast<int>(type));
        return nullptr;
    }

    auto codec = it->second();
    if (!codec || !codec->isAvailable()) {
        LOG_ERROR_FMT("[VideoDecoderFactory] Backend not available: {}", static_cast<int>(type));
        return nullptr;
    }

    return codec;
}

std::vector<std::pair<VideoDecoderBackend, std::string>> VideoDecoderFactory::getAvailableBackends() const {
    std::vector<std::pair<VideoDecoderBackend, std::string>> result;
    for (const auto& [type, creator] : creators_) {
        auto codec = creator();
        if (codec && codec->isAvailable()) {
            result.emplace_back(type, codec->getName());
        }
    }
    return result;
}

bool VideoDecoderFactory::isBackendAvailable(VideoDecoderBackend type) const {
    auto it = creators_.find(type);
    if (it == creators_.end()) return false;
    auto codec = it->second();
    return codec && codec->isAvailable();
}

} // namespace hal
} // namespace ai_stream
