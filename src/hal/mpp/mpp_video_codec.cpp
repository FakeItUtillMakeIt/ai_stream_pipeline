// src/hal/mpp/mpp_video_codec.cpp
// MPP 视频编解码——Rockchip RK3588 多媒体处理平台
#include "mpp_video_codec.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

MppVideoCodec::MppVideoCodec() {
    initialized_ = initMpp();
    if (initialized_) {
        LOG_DEBUG("[MppVideoCodec] Initialized");
    }
}

MppVideoCodec::~MppVideoCodec() {
    cleanup();
    LOG_DEBUG("[MppVideoCodec] Destroyed");
}

bool MppVideoCodec::init(const std::string& codec_name,
                          const uint8_t* extradata,
                          int extradata_size) {
    codec_name_ = codec_name;
    extradata_ = extradata;
    extradata_size_ = extradata_size;

    LOG_INFO_FMT("[MppVideoCodec] Initializing codec: {}", codec_name);
    return initialized_;
}

bool MppVideoCodec::decode(const uint8_t* packet_data, int packet_size,
                            DecodedFrame& frame) {
    if (!initialized_) {
        LOG_ERROR("[MppVideoCodec] Not initialized");
        return false;
    }

    // 实际实现：
    // 1. 创建 MppBuffer 输入包
    // 2. mpp_api_->decode_put_packet()
    // 3. mpp_api_->decode_get_frame()
    // 4. 填充 frame 结构

    LOG_DEBUG_FMT("[MppVideoCodec] decode: input {} bytes", packet_size);
    return true;
}

void MppVideoCodec::release() {
    cleanup();
    LOG_DEBUG("[MppVideoCodec] Released");
}

std::string MppVideoCodec::getName() const {
    return "MPP (Rockchip)";
}

bool MppVideoCodec::isAvailable() const {
#ifdef WITH_RKNN
    return initialized_;
#else
    return false;
#endif
}

bool MppVideoCodec::initMpp() {
    LOG_WARN("[MppVideoCodec] MPP backend is not implemented");
    return false;
}

void MppVideoCodec::cleanup() {
    initialized_ = false;
}

// 注册 MPP 后端到工厂
#ifdef WITH_RKNN
REGISTER_VIDEO_CODEC(VideoCodecBackend::MPP, MppVideoCodec)
#endif

} // namespace hal
} // namespace ai_stream
