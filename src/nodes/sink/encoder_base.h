// src/nodes/sink/encoder_base.h
// 编码与封装编排基类：硬件/软编实现在 HAL（VideoEncoderFactory），
// 本类负责 BGR→YUV 色彩转换与 FLV/MP4/RTMP 容器封装。
#pragma once

#include <string>
#include <memory>
#include <vector>
#include <atomic>

#include "ai_stream/hal/i_video_encoder.h"

struct AVFormatContext;
struct AVCodecContext;
struct AVCodec;
struct AVStream;
struct AVFrame;
struct SwsContext;

namespace ai_stream {
namespace nodes {

/**
 * @brief 编码器基类（HAL 编码 + FFmpeg 封装编排）
 *
 * 编码路径：
 * - HAL 后端（mpp_h264 / ffmpeg_h264 / auto）→ YUV420P 交 IVideoEncoder
 * - HAL 不可用时回退 legacy avcodec 软编路径（行为与历史版本一致）
 */
class EncoderBase {
public:
    EncoderBase() = default;
    virtual ~EncoderBase();

    // 释放编码与输出资源
    virtual void close();

    bool init(const std::string& output_url,
              const std::string& format_name,
              int width, int height,
              int bitrate,
              const std::string& encoder_name = "libx264");

    // 编码单帧（public：VideoRecorder 等外部组件直接调用）
    virtual bool encodeFrame(const uint8_t* data, int width, int height,
                             int step, int64_t pts);
    virtual void flush();

protected:
    virtual bool openOutput(const std::string& url, const std::string& format_name) = 0;
    bool addVideoStream(int width, int height, int bitrate, const std::string& encoder_name);
    bool openVideoCodec();
    bool writeHeader();
    void writeTrailer();

    // 编码实现路径
    bool encodeViaHal(const uint8_t* data, int width, int height, int step, int64_t pts);
    bool encodeLegacy(const uint8_t* data, int width, int height, int step, int64_t pts);
    bool muxPackets(const std::vector<hal::EncodedPacket>& packets, int64_t pts);

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;   // 封装载体（legacy 路径时为实际编码上下文）
    const AVCodec* video_codec_ = nullptr;
    AVStream* video_stream_ = nullptr;
    AVFrame* av_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;

    // HAL 编码后端
    hal::VideoEncoderPtr hal_encoder_;
    std::string configEncoderName_;  // init() 传入的原始编码器名
    bool legacy_codec_opened_ = false;

    int width_ = 0;
    int height_ = 0;
    bool initialized_ = false;
    int64_t next_pts_ = 0;
    std::atomic<bool> closed_{false};
    std::atomic<bool> flushed_{false};
};

/**
 * @brief 文件编码器（MP4 封装）
 */
class FileEncoder : public EncoderBase {
public:
    bool openOutput(const std::string& url, const std::string& format_name) override;
};

/**
 * @brief RTMP 编码器（FLV 封装）
 */
class RTMPEncoder : public EncoderBase {
public:
    bool openOutput(const std::string& url, const std::string& format_name) override;
};

} // namespace nodes
} // namespace ai_stream
