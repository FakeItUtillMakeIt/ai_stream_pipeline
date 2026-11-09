// src/hal/encode/vaapi/vaapi_video_encoder.h
// Intel/AMD VAAPI 硬件编码 IVideoEncoder 后端（h264_vaapi）
// 输入契约与其它 HAL 编码器一致：连续 YUV420P，输出 AnnexB + AnnexB extradata。
#pragma once

#include "ai_stream/hal/i_video_encoder.h"
#include <vector>

struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVBufferRef;
struct SwsContext;

namespace ai_stream {
namespace hal {

class VaapiVideoEncoder : public IVideoEncoder {
public:
    VaapiVideoEncoder() = default;
    ~VaapiVideoEncoder() override;

    bool open(const VideoEncoderConfig& config) override;
    bool encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                std::vector<EncodedPacket>& packets) override;
    bool flush(std::vector<EncodedPacket>& packets) override;
    void close() override;

    const uint8_t* getExtradata(size_t& size) const override {
        size = extradata_.size();
        return extradata_.empty() ? nullptr : extradata_.data();
    }

    std::string getName() const override { return "VAAPI H.264 Encoder"; }
    bool isAvailable() const override;

private:
    bool drainPackets(std::vector<EncodedPacket>& packets);
    void destroy();

    AVCodecContext* codec_ctx_ = nullptr;
    AVFrame* sw_frame_ = nullptr;   // NV12 主机帧
    AVFrame* hw_frame_ = nullptr;   // VAAPI 设备帧
    AVPacket* packet_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    AVBufferRef* hw_frames_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;

    std::vector<uint8_t> extradata_;
    std::vector<std::vector<uint8_t>> staging_frames_;  // 借用至下次 encode()

    int width_ = 0;
    int height_ = 0;
    bool opened_ = false;
};

} // namespace hal
} // namespace ai_stream
