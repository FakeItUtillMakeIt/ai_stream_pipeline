// src/hal/encode/horizon/horizon_video_encoder.h
// Horizon（RDK S100P）VPU 硬件 H.264 编码器——HAL IVideoEncoder 后端
//
// 通过 dlopen 加载 libspcdev.so（sp_init_encoder_module 等），
// 在无该库的编译主机上可构建（运行时 isAvailable()==false，节点回退软编）。
//
// 输入：YUV420P 平面数据（连续 w*h*3/2）
// 输出：AnnexB H.264
#pragma once

#include "ai_stream/hal/i_video_encoder.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class HorizonVideoEncoder : public IVideoEncoder {
public:
    HorizonVideoEncoder();
    ~HorizonVideoEncoder() override;

    bool open(const VideoEncoderConfig& config) override;
    bool encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                std::vector<EncodedPacket>& packets) override;
    bool flush(std::vector<EncodedPacket>& packets) override { (void)packets; return true; }
    void close() override;

    const uint8_t* getExtradata(size_t& size) const override {
        size = extradata_.size();
        return extradata_.empty() ? nullptr : extradata_.data();
    }

    std::string getName() const override { return "Horizon SP H.264 (VPU)"; }
    bool isAvailable() const override;

private:
    void closeUnlocked();  // 内部使用：调用者需已持有 mutex_

    void* obj_ = nullptr;          // sp encoder module handle
    int width_ = 0;
    int height_ = 0;
    int gop_ = 25;

    std::vector<uint8_t> nv12_;       // I420 -> NV12 输入缓冲
    std::vector<uint8_t> stream_;     // 输出码流缓冲
    std::vector<uint8_t> extradata_;  // AnnexB SPS/PPS
    bool got_extradata_ = false;
    bool opened_ = false;
    std::mutex mutex_;
};

} // namespace hal
} // namespace ai_stream
