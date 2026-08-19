// src/hal/nvdec/nvdec_video_codec.h
// NVIDIA NVDEC 硬件视频解码——封装 FFmpeg + CUDA hwaccel 到 HAL 接口
#pragma once

#include "ai_stream/hal/i_video_codec.h"
#include <string>
#include <mutex>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

namespace ai_stream {
namespace hal {

class NvdecVideoCodec : public IVideoCodec {
public:
    NvdecVideoCodec();
    ~NvdecVideoCodec() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;

    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;

    void release() override;

    std::string getName() const override { return "NVDEC (NVIDIA HW)"; }
    bool isAvailable() const override;

private:
    bool initDecoder();
    bool initCudaContext();
    void cleanup();

    std::string codec_name_;
    const uint8_t* extradata_ = nullptr;
    int extradata_size_ = 0;

    AVCodecContext* codec_ctx_ = nullptr;
    AVFrame* frame_ = nullptr;
    AVFrame* hw_frame_ = nullptr;
    AVPacket* packet_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;

    bool initialized_ = false;
    bool cuda_initialized_ = false;
    int device_id_ = 0;
};

} // namespace hal
} // namespace ai_stream
