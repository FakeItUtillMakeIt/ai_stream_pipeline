// src/nodes/sink/encoder_base.cpp
// 编码与封装编排实现：HAL 编码后端 + legacy avcodec 软编兜底
#include "encoder_base.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <cstring>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/frame.h>
#include <libswscale/swscale.h>
}

namespace ai_stream {
namespace nodes {

// ============================================================
// EncoderBase 实现
// ============================================================

EncoderBase::~EncoderBase() {
    close();
}

bool EncoderBase::init(const std::string& output_url,
                       const std::string& format_name,
                       int width, int height,
                       int bitrate,
                       const std::string& encoder_name) {
    width_ = width;
    height_ = height;
    configEncoderName_ = encoder_name;

    LOG_INFO_FMT("[EncoderBase] Initializing: url={}, format={}, {}x{}, {}kbps, encoder={}",
                 output_url, format_name, width, height, bitrate, encoder_name);

    if (!openOutput(output_url, format_name)) {
        LOG_ERROR("[EncoderBase] Failed to open output");
        return false;
    }

    if (!addVideoStream(width, height, bitrate, encoder_name)) {
        LOG_ERROR("[EncoderBase] Failed to add video stream");
        close();
        return false;
    }

    if (!openVideoCodec()) {
        LOG_ERROR("[EncoderBase] Failed to open video codec");
        close();
        return false;
    }

    // 分配帧（YUV420P，编码输入统一格式）
    av_frame_ = av_frame_alloc();
    if (!av_frame_) {
        LOG_ERROR("[EncoderBase] Failed to allocate frame");
        close();
        return false;
    }
    av_frame_->format = codec_ctx_->pix_fmt;
    av_frame_->width = width;
    av_frame_->height = height;
    int ret = av_frame_get_buffer(av_frame_, 0);
    if (ret < 0) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[EncoderBase] Failed to allocate frame buffer: {}", errbuf);
        close();
        return false;
    }

    // BGR24 -> YUV420P 色彩转换（节点侧统一由 BGR Mat 输入）
    sws_ctx_ = sws_getContext(
        width, height, AV_PIX_FMT_BGR24,
        width, height, codec_ctx_->pix_fmt,
        SWS_FAST_BILINEAR, nullptr, nullptr, nullptr
    );
    if (!sws_ctx_) {
        LOG_ERROR("[EncoderBase] Failed to create swscale context");
        close();
        return false;
    }

    if (!writeHeader()) {
        LOG_ERROR("[EncoderBase] Failed to write header");
        close();
        return false;
    }

    initialized_ = true;
    LOG_INFO_FMT("[EncoderBase] Initialized successfully (encoder={})",
                 hal_encoder_ ? hal_encoder_->getName() : std::string("legacy avcodec"));
    return true;
}

bool EncoderBase::addVideoStream(int width, int height, int bitrate,
                                  const std::string& encoder_name) {
    // 编码上下文在此仅作封装载体（codecpar/time_base）；实际编码在 HAL。
    // legacy 路径复用该上下文执行 avcodec_open2。
    if (encoder_name.find("mpp") == std::string::npos) {
        const AVCodec* codec = avcodec_find_encoder_by_name(encoder_name.c_str());
        if (!codec) {
            LOG_WARN_FMT("[EncoderBase] Encoder '{}' not found, trying default H264",
                         encoder_name);
            codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        }
        if (!codec) {
            LOG_ERROR("[EncoderBase] No encoder found");
            return false;
        }
        video_codec_ = codec;
        codec_ctx_ = avcodec_alloc_context3(codec);
        codec_ctx_->codec_id = codec->id;
        LOG_INFO_FMT("[EncoderBase] Using encoder: {}", codec->name);
    } else {
        const AVCodec* codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        codec_ctx_ = avcodec_alloc_context3(codec);
        codec_ctx_->codec_id = AV_CODEC_ID_H264;
    }
    if (!codec_ctx_) {
        LOG_ERROR("[EncoderBase] Failed to allocate codec context");
        return false;
    }

    video_stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!video_stream_) {
        LOG_ERROR("[EncoderBase] Failed to create stream");
        return false;
    }
    video_stream_->id = fmt_ctx_->nb_streams - 1;

    codec_ctx_->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx_->bit_rate = bitrate * 1000;
    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = (AVRational){1, 25};
    codec_ctx_->framerate = (AVRational){25, 1};
    codec_ctx_->gop_size = 12;
    codec_ctx_->max_b_frames = 2;
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    video_stream_->time_base = codec_ctx_->time_base;

    LOG_INFO_FMT("[EncoderBase] Video stream added: {}x{} @ {}kbps", width, height, bitrate);
    return true;
}

bool EncoderBase::openVideoCodec() {
    // 1. 创建 HAL 编码后端
    // 名称映射：mpp_h264 → mpp；auto → auto；其余（libx264/nvenc...）→ ffmpeg 后端
    std::string backend = "auto";
    hal::VideoEncoderConfig cfg;
    cfg.width = width_;
    cfg.height = height_;
    cfg.bitrate_kbps = static_cast<int>(codec_ctx_->bit_rate / 1000);
    cfg.fps = codec_ctx_->time_base.den;
    cfg.gop = codec_ctx_->gop_size;
    cfg.codec_name = video_codec_ ? video_codec_->name : "libx264";

    if (configEncoderName_.find("mpp") != std::string::npos) {
        backend = "mpp_h264";
    } else if (configEncoderName_ != "auto") {
        backend = "ffmpeg_h264";
    }

    hal_encoder_ = hal::VideoEncoderFactory::instance().create(backend);
    if (hal_encoder_ && hal_encoder_->isAvailable() && hal_encoder_->open(cfg)) {
        // 序列头交给封装载体（FLV/MP4 需要 AVCC extradata）
        size_t ed_size = 0;
        const uint8_t* ed = hal_encoder_->getExtradata(ed_size);
        if (ed && ed_size > 0) {
            codec_ctx_->extradata = static_cast<uint8_t*>(
                av_mallocz(ed_size + AV_INPUT_BUFFER_PADDING_SIZE));
            memcpy(codec_ctx_->extradata, ed, ed_size);
            codec_ctx_->extradata_size = static_cast<int>(ed_size);
        }
        if (avcodec_parameters_from_context(video_stream_->codecpar, codec_ctx_) < 0) {
            LOG_ERROR("[EncoderBase] avcodec_parameters_from_context failed");
            return false;
        }
        LOG_INFO("[EncoderBase] HAL encoder active");
        return true;
    }
    LOG_WARN_FMT("[EncoderBase] HAL encoder '{}' unavailable, falling back to legacy avcodec",
                 backend);
    hal_encoder_.reset();

    // 2. legacy：avcodec 软编直驱
    if (!video_codec_) {
        LOG_ERROR("[EncoderBase] Codec not resolved");
        return false;
    }
    int ret = avcodec_open2(codec_ctx_, video_codec_, nullptr);
    if (ret < 0) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[EncoderBase] Failed to open codec: {}", errbuf);
        return false;
    }
    if (avcodec_parameters_from_context(video_stream_->codecpar, codec_ctx_) < 0) {
        LOG_ERROR("[EncoderBase] avcodec_parameters_from_context failed");
        return false;
    }
    legacy_codec_opened_ = true;
    LOG_INFO("[EncoderBase] Legacy avcodec encoder active");
    return true;
}

bool EncoderBase::writeHeader() {
    if (!fmt_ctx_) {
        LOG_ERROR("[EncoderBase] Format context is null");
        return false;
    }
    av_dump_format(fmt_ctx_, 0, fmt_ctx_->url, 1);
    int ret = avformat_write_header(fmt_ctx_, nullptr);
    if (ret < 0) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[EncoderBase] Failed to write header: {} (ret={})", errbuf, ret);
        return false;
    }
    LOG_INFO("[EncoderBase] Header written successfully");
    return true;
}

bool EncoderBase::encodeFrame(const uint8_t* data, int width, int height,
                              int step, int64_t pts) {
    if (!data) {
        LOG_ERROR("[EncoderBase] Invalid input data");
        return false;
    }
    if (width <= 0 || height <= 0 || step <= 0) {
        LOG_ERROR("[EncoderBase] Invalid input size");
        return false;
    }
    if (!initialized_) {
        LOG_ERROR("[EncoderBase] Not initialized");
        return false;
    }
    if (hal_encoder_) {
        return encodeViaHal(data, width, height, step, pts);
    }
    return encodeLegacy(data, width, height, step, pts);
}

bool EncoderBase::encodeViaHal(const uint8_t* data, int width, int height,
                               int step, int64_t pts) {
    if (av_frame_make_writable(av_frame_) < 0) {
        LOG_ERROR("[EncoderBase] av_frame_make_writable failed");
        return false;
    }
    const uint8_t* src_data[1] = {data};
    const int src_linesize[1] = {step};
    sws_scale(sws_ctx_, src_data, src_linesize, 0, height,
              av_frame_->data, av_frame_->linesize);

    // 三平面 → 连续 YUV420P
    const int w = av_frame_->width;
    const int h = av_frame_->height;
    std::vector<uint8_t> yuv(static_cast<size_t>(w) * h * 3 / 2);
    uint8_t* dst_y = yuv.data();
    uint8_t* dst_u = dst_y + static_cast<size_t>(w) * h;
    uint8_t* dst_v = dst_u + static_cast<size_t>(w) * h / 4;
    for (int y = 0; y < h; ++y) {
        memcpy(dst_y + static_cast<size_t>(y) * w,
               av_frame_->data[0] + static_cast<size_t>(y) * av_frame_->linesize[0], w);
    }
    for (int y = 0; y < h / 2; ++y) {
        memcpy(dst_u + static_cast<size_t>(y) * (w / 2),
               av_frame_->data[1] + static_cast<size_t>(y) * av_frame_->linesize[1], w / 2);
        memcpy(dst_v + static_cast<size_t>(y) * (w / 2),
               av_frame_->data[2] + static_cast<size_t>(y) * av_frame_->linesize[2], w / 2);
    }

    std::vector<hal::EncodedPacket> packets;
    if (!hal_encoder_->encode(yuv.data(), yuv.size(), pts, packets)) {
        LOG_ERROR("[EncoderBase] HAL encode failed");
        return false;
    }
    return muxPackets(packets, pts);
}

bool EncoderBase::encodeLegacy(const uint8_t* data, int width, int height,
                               int step, int64_t pts) {
    if (av_frame_make_writable(av_frame_) < 0) {
        LOG_ERROR("[EncoderBase] av_frame_make_writable failed");
        return false;
    }
    const uint8_t* src_data[1] = {data};
    const int src_linesize[1] = {step};
    sws_scale(sws_ctx_, src_data, src_linesize, 0, height,
              av_frame_->data, av_frame_->linesize);
    av_frame_->pts = pts;

    int ret = avcodec_send_frame(codec_ctx_, av_frame_);
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[EncoderBase] Failed to send frame: {}", errbuf);
        return false;
    }

    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        LOG_ERROR("[EncoderBase] Failed to allocate packet");
        return false;
    }
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx_, pkt);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) {
            char errbuf[256] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[EncoderBase] Failed to receive packet: {}", errbuf);
            av_packet_free(&pkt);
            return false;
        }
        av_packet_rescale_ts(pkt, codec_ctx_->time_base, video_stream_->time_base);
        pkt->stream_index = video_stream_->index;
        ret = av_interleaved_write_frame(fmt_ctx_, pkt);
        if (ret < 0) {
            char errbuf[256] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[EncoderBase] Failed to write frame: {}", errbuf);
            av_packet_unref(pkt);
            av_packet_free(&pkt);
            return false;
        }
        av_packet_unref(pkt);
    }
    av_packet_free(&pkt);
    return true;
}

bool EncoderBase::muxPackets(const std::vector<hal::EncodedPacket>& packets, int64_t /*pts*/) {
    for (const auto& p : packets) {
        AVPacket* pkt = av_packet_alloc();
        if (!pkt) return false;
        if (av_new_packet(pkt, static_cast<int>(p.size)) < 0) {
            av_packet_free(&pkt);
            return false;
        }
        memcpy(pkt->data, p.data, p.size);
        pkt->pts = p.pts;
        pkt->dts = p.dts;
        if (p.keyframe) pkt->flags |= AV_PKT_FLAG_KEY;
        av_packet_rescale_ts(pkt, codec_ctx_->time_base, video_stream_->time_base);
        pkt->stream_index = video_stream_->index;

        int ret = av_interleaved_write_frame(fmt_ctx_, pkt);
        av_packet_free(&pkt);
        if (ret < 0) {
            char errbuf[256] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[EncoderBase] Failed to write frame: {}", errbuf);
            return false;
        }
    }
    return true;
}

void EncoderBase::flush() {
    if (!initialized_ || !fmt_ctx_ || flushed_.exchange(true)) return;

    LOG_INFO("[EncoderBase] Flushing encoder...");

    // legacy avcodec 路径需要 drain
    if (legacy_codec_opened_ && codec_ctx_) {
        int ret = avcodec_send_frame(codec_ctx_, nullptr);
        if (ret < 0 && ret != AVERROR_EOF) {
            LOG_WARN_FMT("[EncoderBase] Failed to send frame: {}", ret);
        }
        AVPacket* pkt = av_packet_alloc();
        if (pkt) {
            while (true) {
                ret = avcodec_receive_packet(codec_ctx_, pkt);
                if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF || ret < 0) break;
                av_packet_rescale_ts(pkt, codec_ctx_->time_base, video_stream_->time_base);
                pkt->stream_index = video_stream_->index;
                av_interleaved_write_frame(fmt_ctx_, pkt);
                av_packet_unref(pkt);
            }
            av_packet_free(&pkt);
        }
    }

    if (hal_encoder_) {
        hal_encoder_->close();
    }

    if (fmt_ctx_ && fmt_ctx_->pb) {
        int ret = av_write_trailer(fmt_ctx_);
        if (ret < 0) {
            LOG_WARN_FMT("[EncoderBase] Failed to write trailer: {}", ret);
        }
    }
    LOG_INFO("[EncoderBase] Flush complete");
}

void EncoderBase::writeTrailer() {
    if (fmt_ctx_) {
        av_write_trailer(fmt_ctx_);
    }
}

void EncoderBase::close() {
    if (closed_.exchange(true)) return;

    if (initialized_) {
        flush();
        initialized_ = false;
    }

    // legacy 路径确保编码器输出已读完
    if (legacy_codec_opened_ && codec_ctx_) {
        int ret;
        AVPacket* pkt = av_packet_alloc();
        if (pkt) {
            while ((ret = avcodec_receive_packet(codec_ctx_, pkt)) >= 0) {
                av_packet_unref(pkt);
            }
            av_packet_free(&pkt);
        }
    }

    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (av_frame_) {
        av_frame_free(&av_frame_);
    }
    if (hal_encoder_) {
        hal_encoder_->close();
        hal_encoder_.reset();
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
        if (fmt_ctx_->pb && !(fmt_ctx_->flags & AVFMT_NOFILE)) {
            avio_flush(fmt_ctx_->pb);
            avio_closep(&fmt_ctx_->pb);
        }
        avformat_free_context(fmt_ctx_);
        fmt_ctx_ = nullptr;
    }
    LOG_INFO("[EncoderBase] Closed");
}

// ============================================================
// FileEncoder 实现
// ============================================================

bool FileEncoder::openOutput(const std::string& url, const std::string& format_name) {
    int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "mp4", url.c_str());
    if (ret < 0 || !fmt_ctx_) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[FileEncoder] Failed to allocate output context: {} (ret={})", errbuf, ret);
        return false;
    }
    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&fmt_ctx_->pb, url.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            char errbuf[256] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[FileEncoder] Failed to open output file: {} (ret={})", errbuf, ret);
            avformat_free_context(fmt_ctx_);
            fmt_ctx_ = nullptr;
            return false;
        }
    }
    LOG_INFO_FMT("[FileEncoder] Output opened: {}", url);
    return true;
}

// ============================================================
// RTMPEncoder 实现
// ============================================================

bool RTMPEncoder::openOutput(const std::string& url, const std::string& format_name) {
    int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, "flv", url.c_str());
    if (ret < 0 || !fmt_ctx_) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[RTMPEncoder] Failed to allocate output context: {} (ret={})", errbuf, ret);
        return false;
    }
    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&fmt_ctx_->pb, url.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            char errbuf[256] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[RTMPEncoder] Failed to open output: {} (ret={})", errbuf, ret);
            avformat_free_context(fmt_ctx_);
            fmt_ctx_ = nullptr;
            return false;
        }
    }
    LOG_INFO_FMT("[RTMPEncoder] Output opened: {}", url);
    return true;
}

} // namespace nodes
} // namespace ai_stream
