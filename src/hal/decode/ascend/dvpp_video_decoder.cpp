// src/hal/ascend/dvpp_video_decoder.cpp
// DVPP 视频编解码——华为 Ascend 数字视觉预处理引擎
#include "dvpp_video_decoder.h"
#include "ai_stream/hal/video_decoder_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

DvppVideoDecoder::DvppVideoDecoder() {
    initialized_ = initDvpp();
    if (initialized_) {
        LOG_DEBUG("[DvppVideoDecoder] Initialized");
    }
}

DvppVideoDecoder::~DvppVideoDecoder() {
    cleanup();
    LOG_DEBUG("[DvppVideoDecoder] Destroyed");
}

bool DvppVideoDecoder::init(const std::string& codec_name,
                           const uint8_t* extradata,
                           int extradata_size) {
    codec_name_ = codec_name;
    extradata_ = extradata;
    extradata_size_ = extradata_size;

    LOG_INFO_FMT("[DvppVideoDecoder] Initializing codec: {}", codec_name);
    return initialized_;
}

bool DvppVideoDecoder::decode(const uint8_t* packet_data, int packet_size,
                             DecodedFrame& frame) {
    if (!initialized_) {
        LOG_ERROR("[DvppVideoDecoder] Not initialized");
        return false;
    }

    // 实际实现：
    // 1. 创建输入流描述
    // 2. 创建输出帧描述
    // 3. 调用 vdec 解码

    LOG_DEBUG_FMT("[DvppVideoDecoder] decode: input {} bytes", packet_size);
    return true;
}

void DvppVideoDecoder::release() {
    cleanup();
    LOG_DEBUG("[DvppVideoDecoder] Released");
}

std::string DvppVideoDecoder::getName() const {
    return "DVPP (Ascend)";
}

bool DvppVideoDecoder::isAvailable() const {
#ifdef WITH_ASCEND
    return initialized_;
#else
    return false;
#endif
}

bool DvppVideoDecoder::initDvpp() {
    LOG_WARN("[DvppVideoDecoder] DVPP backend is not implemented");
    return false;
}

void DvppVideoDecoder::cleanup() {
    initialized_ = false;
}

// 注册 DVPP 后端到工厂
#ifdef WITH_ASCEND
REGISTER_VIDEO_DECODER(VideoDecoderBackend::DVPP, DvppVideoDecoder)
#endif

} // namespace hal
} // namespace ai_stream
