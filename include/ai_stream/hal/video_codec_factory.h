// include/ai_stream/hal/video_codec_factory.h
// 视频编解码工厂——根据编译选项和运行时配置创建具体后端
#pragma once

#include "ai_stream/hal/i_video_codec.h"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace ai_stream {
namespace hal {

/**
 * @brief 视频编解码后端类型
 */
enum class VideoCodecBackend {
    AUTO,       // 自动选择可用后端
    NVDEC,      // NVIDIA NVDEC/NVENC
    MPP,        // Rockchip MPP (RK3588)
    DVPP,       // Huawei Ascend DVPP
    FFMPEG      // FFmpeg 软件编解码
};

/**
 * @brief 视频编解码工厂
 */
class VideoCodecFactory {
public:
    using Creator = std::function<VideoCodecPtr()>;

    static VideoCodecFactory& instance();

    void registerBackend(VideoCodecBackend type, Creator creator);
    VideoCodecPtr create(VideoCodecBackend type = VideoCodecBackend::AUTO);
    std::vector<std::pair<VideoCodecBackend, std::string>> getAvailableBackends() const;
    bool isBackendAvailable(VideoCodecBackend type) const;

private:
    VideoCodecFactory() = default;
    std::unordered_map<VideoCodecBackend, Creator> creators_;
};

#define REGISTER_VIDEO_CODEC(backend_type, class_name) \
    static struct _VideoCodecRegistrar_##class_name { \
        _VideoCodecRegistrar_##class_name() { \
            VideoCodecFactory::instance().registerBackend( \
                backend_type, []() -> VideoCodecPtr { \
                    return std::make_unique<class_name>(); \
                }); \
        } \
    } _video_codec_registrar_##class_name;

} // namespace hal
} // namespace ai_stream
