// src/hal/ffmpeg/ffmpeg_video_encoder.h
// FFmpeg 软件编码 IVideoEncoder 后端（libx264 / h264_nvenc 等通用路径）
// 输出契约与 MppVideoEncoder 一致：AnnexB 包 + AVCC extradata
#pragma once

#include "ai_stream/hal/i_video_encoder.h"

struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace ai_stream {
namespace hal {

class FFmpegVideoEncoder : public IVideoEncoder {
public:
    FFmpegVideoEncoder() = default;
    ~FFmpegVideoEncoder() override;

    bool open(const VideoEncoderConfig& config) override;
    bool encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                std::vector<EncodedPacket>& packets) override;
    void close() override;

    const uint8_t* getExtradata(size_t& size) const override {
        size = extradata_.size();
        return extradata_.empty() ? nullptr : extradata_.data();
    }

    std::string getName() const override { return "FFmpeg Video Encoder (" + codec_name_ + ")"; }
    bool isAvailable() const override { return true; }

private:
    void destroy();

    AVCodecContext* codec_ctx_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    std::vector<uint8_t> extradata_;             // AVCC
    std::vector<std::vector<uint8_t>> staging_frames_;  // 包数据（借用至下次 encode）
    std::string codec_name_ = "libx264";
    bool opened_ = false;
};

} // namespace hal
} // namespace ai_stream
