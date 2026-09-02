// src/nodes/sink/encoder_base.h
#pragma once

#include <string>
#include <memory>
#include <atomic>

struct AVFormatContext;
struct AVCodecContext;
struct AVCodec;
struct AVStream;
struct AVFrame;
struct SwsContext;

namespace ai_stream {
namespace nodes {

/**
 * @brief FFmpeg 编码器基类
 */
class EncoderBase {
public:
    EncoderBase() = default;
    virtual ~EncoderBase();

    // 释放编码与输出资源；硬件后端需先释放自身再调基类逻辑
    virtual void close();

    bool init(const std::string& output_url,
              const std::string& format_name,
              int width, int height,
              int bitrate,
              const std::string& encoder_name = "libx264");

    // 编码单帧：默认 FFmpeg 软编码路径；MPP 等硬件后端覆写（public：
    // VideoRecorder 等外部组件直接调用）
    virtual bool encodeFrame(const uint8_t* data, int width, int height,
                             int step, int64_t pts);
    virtual void flush();

protected:
    virtual bool openOutput(const std::string& url, const std::string& format_name) = 0;
    bool addVideoStream(int width, int height, int bitrate, const std::string& encoder_name);
    virtual bool openVideoCodec();
    bool writeHeader();
    bool writeFrame(const uint8_t* data, int width, int height, int step, int64_t pts);
    void writeTrailer();

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    const AVCodec* video_codec_ = nullptr;  // addVideoStream 解析到的编码器，openVideoCodec 直接使用
    AVStream* video_stream_ = nullptr;
    AVFrame* av_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    
    int width_ = 0;
    int height_ = 0;
    bool initialized_ = false;
    int64_t next_pts_ = 0;
    std::atomic<bool> closed_{false};
    std::atomic<bool> flushed_{false};
};

/**
 * @brief 文件编码器
 */
class FileEncoder : public EncoderBase {
public:
    bool openOutput(const std::string& url, const std::string& format_name) override;
};

/**
 * @brief RTMP 编码器
 */
class RTMPEncoder : public EncoderBase {
public:
    bool openOutput(const std::string& url, const std::string& format_name) override;
};

} // namespace nodes
} // namespace ai_stream