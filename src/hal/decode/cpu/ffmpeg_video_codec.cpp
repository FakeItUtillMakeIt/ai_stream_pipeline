// src/hal/ffmpeg/ffmpeg_video_codec.cpp
// FFmpeg 软件视频编解码——通用 fallback 后端
#include "ffmpeg_video_codec.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace ai_stream {
namespace hal {

FFmpegVideoCodec::FFmpegVideoCodec() {
    LOG_DEBUG("[FFmpegVideoCodec] Constructor");
}

FFmpegVideoCodec::~FFmpegVideoCodec() {
    cleanup();
    LOG_DEBUG("[FFmpegVideoCodec] Destroyed");
}

bool FFmpegVideoCodec::init(const std::string& codec_name,
                             const uint8_t* extradata,
                             int extradata_size) {
    codec_name_ = codec_name;
    extradata_ = extradata;
    extradata_size_ = extradata_size;

    LOG_INFO_FMT("[FFmpegVideoCodec] Initializing codec: {}", codec_name);
    initialized_ = initDecoder();
    return initialized_;
}

bool FFmpegVideoCodec::decode(const uint8_t* packet_data, int packet_size,
                               DecodedFrame& frame) {
    if (!initialized_ || !codec_ctx_) {
        LOG_ERROR("[FFmpegVideoCodec] Not initialized");
        return false;
    }

    // 使用 av_packet_ref 安全复制 packet 数据，避免 const_cast 导致 UB
    av_packet_unref(packet_);
    packet_->data = static_cast<uint8_t*>(av_malloc(packet_size + AV_INPUT_BUFFER_PADDING_SIZE));
    if (!packet_->data) {
        LOG_ERROR("[FFmpegVideoCodec] Failed to allocate packet buffer");
        return false;
    }
    std::memcpy(packet_->data, packet_data, packet_size);
    packet_->size = packet_size;

    // 发送 packet 到解码器
    int ret = avcodec_send_packet(codec_ctx_, packet_);
    av_freep(&packet_->data);  // 释放临时缓冲区
    if (ret < 0) {
        LOG_ERROR_FMT("[FFmpegVideoCodec] avcodec_send_packet failed: {}", ret);
        return false;
    }

    // 接收解码后的帧
    ret = avcodec_receive_frame(codec_ctx_, frame_);
    if (ret < 0) {
        LOG_ERROR_FMT("[FFmpegVideoCodec] avcodec_receive_frame failed: {}", ret);
        return false;
    }

    // 填充输出帧
    frame.data = frame_->data[0];
    frame.width = codec_ctx_->width;
    frame.height = codec_ctx_->height;
    frame.pitch = frame_->linesize[0];
    frame.format = codec_ctx_->pix_fmt;
    frame.owns_data = false;

    return true;
}

void FFmpegVideoCodec::release() {
    cleanup();
    LOG_DEBUG("[FFmpegVideoCodec] Released");
}

bool FFmpegVideoCodec::isAvailable() const {
    // FFmpeg 始终可用
    return true;
}

bool FFmpegVideoCodec::initDecoder() {
    AVCodecID codec_id = AV_CODEC_ID_NONE;

    if (codec_name_ == "h264" || codec_name_ == "H264") {
        codec_id = AV_CODEC_ID_H264;
    } else if (codec_name_ == "h265" || codec_name_ == "hevc" || codec_name_ == "H265") {
        codec_id = AV_CODEC_ID_HEVC;
    } else if (codec_name_ == "vp9" || codec_name_ == "VP9") {
        codec_id = AV_CODEC_ID_VP9;
    } else if (codec_name_ == "av1" || codec_name_ == "AV1") {
        codec_id = AV_CODEC_ID_AV1;
    }

    const AVCodec* codec = nullptr;
    if (codec_id != AV_CODEC_ID_NONE) {
        codec = avcodec_find_decoder(codec_id);
    } else {
        codec = avcodec_find_decoder_by_name(codec_name_.c_str());
    }

    if (!codec) {
        // 默认 H.264
        codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    }
    if (!codec) {
        LOG_ERROR_FMT("[FFmpegVideoCodec] Codec not found: {}", codec_name_);
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        LOG_ERROR("[FFmpegVideoCodec] Failed to allocate codec context");
        return false;
    }

    if (extradata_ && extradata_size_ > 0) {
        codec_ctx_->extradata = static_cast<uint8_t*>(av_malloc(extradata_size_ + AV_INPUT_BUFFER_PADDING_SIZE));
        std::memcpy(codec_ctx_->extradata, extradata_, extradata_size_);
        codec_ctx_->extradata_size = extradata_size_;
    }

    int ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        LOG_ERROR_FMT("[FFmpegVideoCodec] Failed to open codec: {}", ret);
        return false;
    }

    frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();

    return true;
}

void FFmpegVideoCodec::cleanup() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (packet_) {
        av_packet_free(&packet_);
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    initialized_ = false;
}

// 注册 FFmpeg 后端到工厂（始终可用）
REGISTER_VIDEO_CODEC(VideoCodecBackend::FFMPEG, FFmpegVideoCodec)

} // namespace hal
} // namespace ai_stream
