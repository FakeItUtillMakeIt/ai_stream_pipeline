// src/hal/mpp/mpp_video_codec.h
// MPP 视频编解码——Rockchip RK3588 多媒体处理平台
#pragma once

#include "ai_stream/hal/i_video_codec.h"
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief MPP 视频编解码
 *
 * 使用 Rockchip MPP (Media Process Platform) 进行硬件视频编解码：
 * - 解码：H.264, H.265/HEVC, VP9, AV1
 * - 编码：H.264, H.265/HEVC
 *
 * RK3588 支持 8K@60fps 解码和 8K@30fps 编码。
 */
class MppVideoCodec : public IVideoCodec {
public:
    MppVideoCodec();
    ~MppVideoCodec() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;

    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;

    void release() override;

    std::string getName() const override { return "MPP (Rockchip)"; }
    bool isAvailable() const override;

private:
    bool initMpp();
    void cleanup();

    std::string codec_name_;
    const uint8_t* extradata_ = nullptr;
    int extradata_size_ = 0;
    bool initialized_ = false;
    void* mpp_ctx_ = nullptr;
    void* mpp_api_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
