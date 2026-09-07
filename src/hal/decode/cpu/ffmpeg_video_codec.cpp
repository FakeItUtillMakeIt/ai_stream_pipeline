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

    // 软件解码器输出多为平面格式（YUV420P 等），而下游 decode 节点期望
    // 单平面 BGR（或 NV12 的 Y+UV）。这里统一转换为 BGR24，只需一次
    // swscale，解码节点直接封装为 cv::Mat，避免二次转换拖慢软件解码。
    AVFrame* out = frame_;
    if (frame_->format != AV_PIX_FMT_BGR24) {
        if (!ensureBgrConverter(frame_->width, frame_->height, frame_->format)) {
            return false;
        }
        sws_scale(sws_ctx_,
                  (const uint8_t* const*)frame_->data, frame_->linesize,
                  0, frame_->height,
                  bgr_frame_->data, bgr_frame_->linesize);
        out = bgr_frame_;
    }

    // 填充输出帧
    frame.data = out->data[0];
    frame.width = out->width;
    frame.height = out->height;
    frame.pitch = out->linesize[0];
    frame.data_uv = nullptr;
    frame.pitch_uv = 0;
    frame.format = AV_PIX_FMT_BGR24;
    frame.owns_data = false;

    return true;
}

void FFmpegVideoCodec::release() {
    cleanup();
    LOG_DEBUG("[FFmpegVideoCodec] Released");
}

bool FFmpegVideoCodec::ensureBgrConverter(int width, int height, int src_format) {
    if (src_format == AV_PIX_FMT_BGR24) {
        return true;
    }

    if (bgr_frame_ &&
        bgr_frame_->width == width && bgr_frame_->height == height &&
        bgr_frame_->format == AV_PIX_FMT_BGR24) {
        return true;
    }

    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (bgr_frame_) {
        av_frame_free(&bgr_frame_);
    }

    bgr_frame_ = av_frame_alloc();
    if (!bgr_frame_) {
        LOG_ERROR("[FFmpegVideoCodec] Failed to allocate BGR frame");
        return false;
    }
    bgr_frame_->format = AV_PIX_FMT_BGR24;
    bgr_frame_->width = width;
    bgr_frame_->height = height;
    if (av_frame_get_buffer(bgr_frame_, 32) < 0) {
        LOG_ERROR("[FFmpegVideoCodec] Failed to allocate BGR frame buffer");
        return false;
    }

    sws_ctx_ = sws_getContext(width, height, static_cast<AVPixelFormat>(src_format),
                              width, height, AV_PIX_FMT_BGR24,
                              SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        LOG_ERROR("[FFmpegVideoCodec] Failed to create sws context");
        return false;
    }
    return true;
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
    if (bgr_frame_) {
        av_frame_free(&bgr_frame_);
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
