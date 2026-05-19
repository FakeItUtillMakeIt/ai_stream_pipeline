// src/nodes/sink/encoder_base.h
#pragma once

#include <string>
#include <memory>
#include <atomic>

struct AVFormatContext;
struct AVCodecContext;
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

    bool init(const std::string& output_url, 
              const std::string& format_name,
              int width, int height, 
              int bitrate, 
              const std::string& encoder_name = "libx264");
    
    bool encodeFrame(const uint8_t* data, int width, int height, int step, int64_t pts);
    void flush();
    void close();

protected:
    virtual bool openOutput(const std::string& url, const std::string& format_name) = 0;
    bool addVideoStream(int width, int height, int bitrate, const std::string& encoder_name);
    bool openVideoCodec();
    bool writeHeader();
    bool writeFrame(const uint8_t* data, int width, int height, int step, int64_t pts);
    void writeTrailer();

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVStream* video_stream_ = nullptr;
    AVFrame* av_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    
    int width_ = 0;
    int height_ = 0;
    bool initialized_ = false;
    int64_t next_pts_ = 0;
    std::atomic<bool> closed_{false}; 
};

/**
 * @brief 文件编码器
 */
class FileEncoder : public EncoderBase {
protected:
    bool openOutput(const std::string& url, const std::string& format_name) override;
};

/**
 * @brief RTMP 编码器
 */
class RTMPEncoder : public EncoderBase {
protected:
    bool openOutput(const std::string& url, const std::string& format_name) override;
};

} // namespace nodes
} // namespace ai_stream