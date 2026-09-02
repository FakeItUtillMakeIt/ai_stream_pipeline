// src/nodes/sink/mpp_encoder.h
// MPP 硬编码适配层——把 HAL VideoEncoderFactory 的 mpp_h264 后端接入
// EncoderBase 的 mux 流程（节点层只负责容器封装，硬件代码在 HAL）。
//
// 流程：init 时先建 FFmpeg 输出/流（codecpar 用 MPP 序列头填充），
// encodeFrame 将 BGR → YUV420P（复用基类 sws）后交给 HAL 编码器，
// 输出 AnnexB 包转 FFmpeg AVPacket 写入容器。
#pragma once

#include "encoder_base.h"
#include "ai_stream/hal/i_video_encoder.h"

namespace ai_stream {
namespace nodes {

class MppEncoder : public EncoderBase {
public:
    ~MppEncoder() override;

protected:
    bool openOutput(const std::string& url, const std::string& format_name) override;
    bool openVideoCodec() override;
    bool encodeFrame(const uint8_t* data, int width, int height,
                     int step, int64_t pts) override;
    void close() override;
    void flush() override;

private:
    hal::VideoEncoderPtr hal_encoder_;
    int64_t frame_count_ = 0;
};

} // namespace nodes
} // namespace ai_stream
