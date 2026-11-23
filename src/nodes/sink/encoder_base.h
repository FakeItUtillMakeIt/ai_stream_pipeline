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

    // 复用已打开的编码器，切换到新的输出文件：
    // 结束上一个文件（若仍打开）→ 新建封装上下文/流 → 写文件头。
    // 编码器上下文、HAL 后端、颜色转换上下文均保持复用。
    bool startFile(const std::string& output_url, const std::string& format_name);

    // 结束当前输出文件：排空编码器、写 trailer、释放封装上下文；编码器保持打开。
    void finishFile();

    bool isInitialized() const { return initialized_; }

    // 编码单帧（public：VideoRecorder 等外部组件直接调用）
    virtual bool encodeFrame(const uint8_t* data, int width, int height,
                             int step, int64_t pts);
    virtual void flush();

protected:
    virtual bool openOutput(const std::string& url, const std::string& format_name) = 0;
    bool openCodec(int width, int height, int bitrate, const std::string& encoder_name);
    bool createCodecContext(int width, int height, int bitrate, const std::string& encoder_name);
    bool addStream();
    bool openVideoCodec();
    bool writeHeader();
    void closeMuxer();

    // 编码实现路径
    bool encodeViaHal(const uint8_t* data, int width, int height, int step, int64_t pts);
    bool encodeLegacy(const uint8_t* data, int width, int height, int step, int64_t pts);
    // 输入尺寸与编码尺寸不一致时按需创建 BGR→YUV 缩放上下文
    SwsContext* swsForInput(int width, int height);
    bool muxPackets(const std::vector<hal::EncodedPacket>& packets, int64_t pts);

    AVFormatContext* fmt_ctx_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;   // 封装载体（legacy 路径时为实际编码上下文）
    const AVCodec* video_codec_ = nullptr;
    AVStream* video_stream_ = nullptr;
    AVFrame* av_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    SwsContext* scale_sws_ctx_ = nullptr;   // 源尺寸 != 输出尺寸时使用
    int scale_src_width_ = 0;
    int scale_src_height_ = 0;

    // HAL 编码后端
    hal::VideoEncoderPtr hal_encoder_;
    std::string configEncoderName_;  // init() 传入的原始编码器名
    bool legacy_codec_opened_ = false;

    int width_ = 0;
    int height_ = 0;
    bool initialized_ = false;   // 编码器（codec/HAL/帧缓冲）已就绪
    bool file_open_ = false;     // 当前封装输出已打开并写入文件头
    std::atomic<bool> keyframe_requested_{false};
    int64_t file_pts_base_ = -1; // 当前文件首个包的 dts（用于时间戳归零）
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
