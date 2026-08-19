// src/hal/ffmpeg/ffmpeg_video_codec.h
// FFmpeg 软件视频编解码——通用 fallback 后端
#pragma once

#include "ai_stream/hal/i_video_codec.h"
#include <string>

// 前向声明 FFmpeg 类型
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct SwsContext;

namespace ai_stream {
namespace hal {

/**
 * @brief FFmpeg 软件视频编解码
 *
 * 使用 FFmpeg libavcodec 进行软件视频编解码。
 * 作为所有平台的 fallback 后端，支持广泛的编解码格式。
 */
class FFmpegVideoCodec : public IVideoCodec {
public:
    FFmpegVideoCodec();
    ~FFmpegVideoCodec() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;

    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;

    void release() override;

    std::string getName() const override { return "FFmpeg (Software)"; }
    bool isAvailable() const override;

private:
    bool initDecoder();
    void cleanup();

    std::string codec_name_;
    const uint8_t* extradata_ = nullptr;
    int extradata_size_ = 0;

    AVCodecContext* codec_ctx_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    bool initialized_ = false;
};

} // namespace hal
} // namespace ai_stream
