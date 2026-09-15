// include/ai_stream/hal/video_decoder_factory.h
// 视频编解码工厂——根据编译选项和运行时配置创建具体后端
#pragma once

#include "ai_stream/hal/i_video_decoder.h"
#include <functional>
#include <string>
#include <vector>
#include <unordered_map>

namespace ai_stream {
namespace hal {

/**
 * @brief 视频编解码后端类型
 */
enum class VideoDecoderBackend {
    AUTO,       // 自动选择可用后端
    NVDEC,      // NVIDIA NVDEC/NVENC (cuvid，桌面平台)
    NVV4L2,     // NVIDIA Jetson GStreamer nvv4l2 (NvMM/GPU 路径，本机可用)
    V4L2,       // NVIDIA Jetson 裸 V4L2 M2M (/dev/v4l2-nvdec，真实硬件可用)
    MPP,        // Rockchip MPP (RK3588)
    DVPP,       // Huawei Ascend DVPP
    HORIZON,    // Horizon BPU video codec
    FFMPEG      // FFmpeg 软件编解码
};

/**
 * @brief 视频编解码工厂
 */
class VideoDecoderFactory {
public:
    using Creator = std::function<VideoDecoderPtr()>;

    static VideoDecoderFactory& instance();

    void registerBackend(VideoDecoderBackend type, Creator creator);
    VideoDecoderPtr create(VideoDecoderBackend type = VideoDecoderBackend::AUTO);
    std::vector<std::pair<VideoDecoderBackend, std::string>> getAvailableBackends() const;
    bool isBackendAvailable(VideoDecoderBackend type) const;

private:
    VideoDecoderFactory() = default;
    std::unordered_map<VideoDecoderBackend, Creator> creators_;
};

#define REGISTER_VIDEO_DECODER(backend_type, class_name) \
    static struct _VideoDecoderRegistrar_##class_name { \
        _VideoDecoderRegistrar_##class_name() { \
            VideoDecoderFactory::instance().registerBackend( \
                backend_type, []() -> VideoDecoderPtr { \
                    return std::make_unique<class_name>(); \
                }); \
        } \
    } _video_decoder_registrar_##class_name;

} // namespace hal
} // namespace ai_stream
