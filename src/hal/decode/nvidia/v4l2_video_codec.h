// src/hal/decode/nvidia/v4l2_video_codec.h
// Jetson V4L2 NVDEC 硬件解码后端——直接使用 V4L2 M2M API
// （即 NVIDIA Multimedia API 的底层机制，设备节点 /dev/v4l2-nvdec）。
//
// 输入：H.264/H.265/VP9 编码流（AnnexB）；输出：NV12（Y + 交织 UV）。
#pragma once

#include "ai_stream/hal/i_video_codec.h"

#include <cstdint>
#include <deque>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class V4l2VideoCodec : public IVideoCodec {
public:
    V4l2VideoCodec();
    ~V4l2VideoCodec() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;

    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;

    void release() override;

    std::string getName() const override { return "V4L2 NVDEC (Jetson)"; }
    bool isAvailable() const override;

private:
    bool openDecoder();
    bool setupOutputPlane(uint32_t coded_fmt, uint32_t sizeimage);
    // 首个关键帧触发分辨率变更事件后，建立并启动 capture 平面
    bool setupCapturePlane();
    bool waitResolutionChange(int timeout_ms);
    bool feedPacket(const uint8_t* data, int size);
    bool recycleOutputBuffers();
    bool dequeueCaptureFrame(DecodedFrame& frame);

    int fd_ = -1;

    // output 平面（编码码流输入，MMAP 单平面）
    std::vector<void*> out_mmap_;
    std::vector<size_t> out_size_;
    std::deque<uint32_t> out_free_;
    uint32_t out_count_ = 0;

    // capture 平面（解码输出，MMAP 多平面 NV12M/NV12）
    std::vector<std::vector<void*>> cap_mmap_;   // [buf][plane]
    std::vector<std::vector<size_t>> cap_size_;  // [buf][plane] 字节数
    std::deque<uint32_t> cap_free_;
    uint32_t cap_count_ = 0;

    int width_ = 0;
    int height_ = 0;
    bool capture_on_ = false;
    bool initialized_ = false;

    // 帧输出缓冲（decode() 返回后即被 decode 节点拷贝，内部复用）
    std::vector<uint8_t> frame_y_;
    std::vector<uint8_t> frame_uv_;

    std::vector<uint8_t> extradata_;
    bool fed_extradata_ = false;

    std::string codec_name_;
};

} // namespace hal
} // namespace ai_stream