// src/hal/video_encoder_factory.cpp
// VideoEncoderFactory 实现（接口见 i_video_encoder.h）
#include "ai_stream/hal/i_video_encoder.h"

namespace ai_stream {
namespace hal {

VideoEncoderFactory& VideoEncoderFactory::instance() {
    static VideoEncoderFactory factory;
    return factory;
}

void VideoEncoderFactory::registerBackend(const std::string& name, Creator creator) {
    creators_[name] = std::move(creator);
}

VideoEncoderPtr VideoEncoderFactory::create(const std::string& name) {
    // AUTO：硬件可用则优先（Horizon VPU → Rockchip MPP），否则 FFmpeg 软编
    if (name == "auto") {
        auto hz = create("horizon_h264");
        if (hz && hz->isAvailable()) return hz;
        auto mpp = create("mpp_h264");
        if (mpp && mpp->isAvailable()) return mpp;
        return create("ffmpeg_h264");
    }
    auto it = creators_.find(name);
    if (it == creators_.end()) return nullptr;
    return it->second();
}

std::vector<std::string> VideoEncoderFactory::getAvailableBackends() const {
    std::vector<std::string> names;
    for (const auto& [name, _] : creators_) names.push_back(name);
    return names;
}

} // namespace hal
} // namespace ai_stream
