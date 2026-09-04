// src/hal/mpp/mpp_video_encoder.h
// Rockchip MPP 硬件 H.264 编码器——HAL IVideoEncoder 后端
//
// 与 mpp_video_codec（解码侧）保持一致：通过 dlopen 加载
// librockchip_mpp.so，x86 编译主机可构建（运行时不可用则 isAvailable=false，
// 节点回退软件编码）。
// 输入：YUV420P 平面数据；输出：AnnexB H.264。
#pragma once

#include "ai_stream/hal/i_video_encoder.h"
#include "3rd_party/rk_platform/mpp/include/rockchip/rk_mpi.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class MppVideoEncoder : public IVideoEncoder {
public:
    MppVideoEncoder();
    ~MppVideoEncoder() override;

    bool open(const VideoEncoderConfig& config) override;
    bool encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                std::vector<EncodedPacket>& packets) override;
    void close() override;

    const uint8_t* getExtradata(size_t& size) const override {
        size = extradata_.size();
        return extradata_.empty() ? nullptr : extradata_.data();
    }

    std::string getName() const override { return "MPP H.264 (Rockchip VENC)"; }
    bool isAvailable() const override;

private:
    bool initMpp(const VideoEncoderConfig& config);
    bool fetchHeaderSync();  // MPP_ENC_GET_HDR_SYNC → AVCC extradata

    MppCtx ctx_ = nullptr;
    MppApi* mpi_ = nullptr;
    MppEncCfg cfg_ = nullptr;
    MppBufferGroup buf_group_ = nullptr;
    MppBuffer frame_buf_ = nullptr;
    MppFrame frame_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    int gop_ = 25;
    size_t frame_size_ = 0;

    std::vector<uint8_t> extradata_;  // AVCC（长度前缀）格式序列头

    bool opened_ = false;
    std::mutex mutex_;
};

} // namespace hal
} // namespace ai_stream
