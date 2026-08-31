// src/hal/ascend/dvpp_video_codec.cpp
// DVPP 视频编解码——华为 Ascend 数字视觉预处理引擎
#include "dvpp_video_codec.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

DvppVideoCodec::DvppVideoCodec() {
    initialized_ = initDvpp();
    if (initialized_) {
        LOG_DEBUG("[DvppVideoCodec] Initialized");
    }
}

DvppVideoCodec::~DvppVideoCodec() {
    cleanup();
    LOG_DEBUG("[DvppVideoCodec] Destroyed");
}

bool DvppVideoCodec::init(const std::string& codec_name,
                           const uint8_t* extradata,
                           int extradata_size) {
    codec_name_ = codec_name;
    extradata_ = extradata;
    extradata_size_ = extradata_size;

    LOG_INFO_FMT("[DvppVideoCodec] Initializing codec: {}", codec_name);
    return initialized_;
}

bool DvppVideoCodec::decode(const uint8_t* packet_data, int packet_size,
                             DecodedFrame& frame) {
    if (!initialized_) {
        LOG_ERROR("[DvppVideoCodec] Not initialized");
        return false;
    }

    // 实际实现：
    // 1. 创建输入流描述
    // 2. 创建输出帧描述
    // 3. 调用 vdec 解码

    LOG_DEBUG_FMT("[DvppVideoCodec] decode: input {} bytes", packet_size);
    return true;
}

void DvppVideoCodec::release() {
    cleanup();
    LOG_DEBUG("[DvppVideoCodec] Released");
}

std::string DvppVideoCodec::getName() const {
    return "DVPP (Ascend)";
}

bool DvppVideoCodec::isAvailable() const {
#ifdef WITH_ASCEND
    return initialized_;
#else
    return false;
#endif
}

bool DvppVideoCodec::initDvpp() {
    LOG_WARN("[DvppVideoCodec] DVPP backend is not implemented");
    return false;
}

void DvppVideoCodec::cleanup() {
    initialized_ = false;
}

// 注册 DVPP 后端到工厂
#ifdef WITH_ASCEND
REGISTER_VIDEO_CODEC(VideoCodecBackend::DVPP, DvppVideoCodec)
#endif

} // namespace hal
} // namespace ai_stream
