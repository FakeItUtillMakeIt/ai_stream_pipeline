// src/hal/ascend/dvpp_video_codec.h
// DVPP 视频编解码——华为 Ascend 数字视觉预处理引擎
#pragma once

#include "ai_stream/hal/i_video_codec.h"
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief DVPP 视频编解码
 *
 * 使用华为 Ascend DVPP (Digital Video Pre-Processing) 引擎进行：
 * - 解码：JPEG, H.264, H.265/HEVC
 * - 编码：JPEG, H.264, H.265/HEVC
 *
 * 通过 ACL (Ascend Computing Language) API 调用。
 */
class DvppVideoCodec : public IVideoCodec {
public:
    DvppVideoCodec();
    ~DvppVideoCodec() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;

    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;

    void release() override;

    std::string getName() const override { return "DVPP (Ascend)"; }
    bool isAvailable() const override;

private:
    bool initDvpp();
    void cleanup();

    std::string codec_name_;
    const uint8_t* extradata_ = nullptr;
    int extradata_size_ = 0;
    bool initialized_ = false;
    void* dvpp_channel_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
