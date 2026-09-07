// src/hal/encode/nvidia/gst_v4l2_video_encoder.h
// Jetson GStreamer nvv4l2 硬件 H.264 编码后端（IVideoEncoder）。
//
// 输入：packed YUV420P（连续 Y/U/V 三平面，即 I420）；输出：AnnexB H.264。
// 管线：appsrc(I420) → nvvidconv(NVMM NV12) → nvv4l2h264enc → h264parse → appsink
#pragma once

#include "ai_stream/hal/i_video_encoder.h"

#include <cstdint>
#include <string>
#include <vector>

typedef struct _GstElement GstElement;
typedef struct _GstAppSrc GstAppSrc;
typedef struct _GstAppSink GstAppSink;

namespace ai_stream {
namespace hal {

class GstV4l2VideoEncoder : public IVideoEncoder {
public:
    GstV4l2VideoEncoder();
    ~GstV4l2VideoEncoder() override;

    bool open(const VideoEncoderConfig& config) override;
    bool encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                std::vector<EncodedPacket>& packets) override;
    bool flush(std::vector<EncodedPacket>& packets) override;
    void close() override;

    const uint8_t* getExtradata(size_t& size) const override {
        size = extradata_.size();
        return extradata_.empty() ? nullptr : extradata_.data();
    }

    std::string getName() const override { return "GStreamer nvv4l2 (Jetson VENC)"; }
    bool isAvailable() const override;

private:
    bool buildPipeline();
    bool pullPackets(std::vector<EncodedPacket>& packets, int64_t timeout_ms);

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstElement* appsink_ = nullptr;
    GstAppSrc* app_src_ = nullptr;
    GstAppSink* app_sink_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    int fps_ = 25;
    int bitrate_ = 4000;

    std::vector<uint8_t> extradata_;           // AVCC（长度前缀）序列头
    std::vector<std::vector<uint8_t>> out_buffers_; // 编码输出数据（一次 pullPackets 生命周期内有效）
    int64_t next_pts_ = 0;                       // 无时间戳缓冲的 pts 兜底
    int64_t last_pts_ = -1;                      // 已输出最大 pts（保证严格单调）
    bool opened_ = false;
};

} // namespace hal
} // namespace ai_stream