// src/hal/ffmpeg/ffmpeg_video_encoder.cpp
// FFmpeg 软件编码 IVideoEncoder 后端实现——见头文件说明
#include "ffmpeg_video_encoder.h"
#include "ai_stream/hal/i_video_encoder.h"
#include "ai_stream/hal/h264_extradata.h"
#include "3rd_party/log_mgr/log_mgr.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/opt.h>
#include <libavutil/imgutils.h>
}

namespace ai_stream {
namespace hal {

FFmpegVideoEncoder::~FFmpegVideoEncoder() {
    destroy();
}

void FFmpegVideoEncoder::destroy() {
    if (packet_) { av_packet_free(&packet_); }
    if (frame_) { av_frame_free(&frame_); }
    if (codec_ctx_) { avcodec_free_context(&codec_ctx_); }
    opened_ = false;
}

bool FFmpegVideoEncoder::open(const VideoEncoderConfig& config) {
    if (opened_) return true;

    codec_name_ = config.codec_name.empty() ? "libx264" : config.codec_name;
    const AVCodec* codec = avcodec_find_encoder_by_name(codec_name_.c_str());
    if (!codec) {
        LOG_WARN_FMT("[FFmpegVideoEncoder] encoder '{}' not found, trying default H264",
                     codec_name_);
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
        if (!codec) return false;
        codec_name_ = codec->name;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) return false;

    codec_ctx_->codec_id = AV_CODEC_ID_H264;
    codec_ctx_->width = config.width;
    codec_ctx_->height = config.height;
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    codec_ctx_->time_base = AVRational{1, config.fps > 0 ? config.fps : 25};
    codec_ctx_->framerate = AVRational{config.fps > 0 ? config.fps : 25, 1};
    codec_ctx_->gop_size = config.gop > 0 ? config.gop : config.fps;
    codec_ctx_->bit_rate = static_cast<int64_t>(config.bitrate_kbps) * 1000;
    // 序列头独立输出（extradata），帧包不再重复携带 SPS/PPS
    codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    if (codec->id == AV_CODEC_ID_H264 &&
        std::string(codec->name).find("nvenc") == std::string::npos) {
        av_opt_set(codec_ctx_->priv_data, "preset", "fast", 0);
        av_opt_set(codec_ctx_->priv_data, "tune", "zerolatency", 0);
    } else if (std::string(codec->name).find("nvenc") != std::string::npos) {
        codec_ctx_->max_b_frames = 0;
        av_opt_set(codec_ctx_->priv_data, "preset", "p4", 0);
        av_opt_set(codec_ctx_->priv_data, "tune", "ull", 0);
        av_opt_set(codec_ctx_->priv_data, "rc", "cbr", 0);
    }

    if (avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
        LOG_ERROR_FMT("[FFmpegVideoEncoder] avcodec_open2 failed for {}", codec_name_);
        destroy();
        return false;
    }

    // 契约：extradata 为 AnnexB 原样透传；AVCDecoderConfigurationRecord
    // 由 muxer（ff_isom_write_avcc）生成
    if (codec_ctx_->extradata_size > 0) {
        extradata_.assign(codec_ctx_->extradata,
                          codec_ctx_->extradata + codec_ctx_->extradata_size);
    }

    frame_ = av_frame_alloc();
    if (!frame_) { destroy(); return false; }
    frame_->format = AV_PIX_FMT_YUV420P;
    frame_->width = config.width;
    frame_->height = config.height;
    if (av_frame_get_buffer(frame_, 0) < 0) {
        LOG_ERROR("[FFmpegVideoEncoder] frame buffer alloc failed");
        destroy();
        return false;
    }

    packet_ = av_packet_alloc();
    if (!packet_) { destroy(); return false; }

    opened_ = true;
    LOG_INFO_FMT("[FFmpegVideoEncoder] Opened: {} {}x{} @ {}kbps",
                 codec_name_, config.width, config.height, config.bitrate_kbps);
    return true;
}

bool FFmpegVideoEncoder::encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                                std::vector<EncodedPacket>& packets) {
    if (!opened_) return false;
    const size_t need = static_cast<size_t>(codec_ctx_->width) *
                        codec_ctx_->height * 3 / 2;
    if (!yuv420p || size < need) return false;

    if (av_frame_make_writable(frame_) < 0) return false;

    // 连续 YUV420P → 三平面
    const int w = codec_ctx_->width;
    const int h = codec_ctx_->height;
    const uint8_t* src = yuv420p;
    for (int y = 0; y < h; ++y) {
        memcpy(frame_->data[0] + static_cast<size_t>(y) * frame_->linesize[0],
               src + static_cast<size_t>(y) * w, w);
    }
    const uint8_t* u = src + static_cast<size_t>(w) * h;
    const uint8_t* v = u + static_cast<size_t>(w) * h / 4;
    for (int y = 0; y < h / 2; ++y) {
        memcpy(frame_->data[1] + static_cast<size_t>(y) * frame_->linesize[1],
               u + static_cast<size_t>(y) * (w / 2), w / 2);
        memcpy(frame_->data[2] + static_cast<size_t>(y) * frame_->linesize[2],
               v + static_cast<size_t>(y) * (w / 2), w / 2);
    }
    frame_->pts = pts;

    int ret = avcodec_send_frame(codec_ctx_, frame_);
    if (ret < 0 && ret != AVERROR(EAGAIN)) {
        LOG_ERROR_FMT("[FFmpegVideoEncoder] send_frame failed: {}", ret);
        return false;
    }

    staging_frames_.clear();
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx_, packet_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) {
            LOG_ERROR_FMT("[FFmpegVideoEncoder] receive_packet failed: {}", ret);
            return false;
        }
        // 每包独立持有（满足"借用至下次 encode"契约，避免 insert 重分配悬垂）
        staging_frames_.emplace_back(packet_->data, packet_->data + packet_->size);
        auto& buf = staging_frames_.back();
        EncodedPacket p;
        p.data = buf.data();
        p.size = buf.size();
        p.pts = packet_->pts;
        p.dts = packet_->dts;
        p.keyframe = (packet_->flags & AV_PKT_FLAG_KEY) != 0;
        packets.push_back(p);
        av_packet_unref(packet_);
    }
    return true;
}

void FFmpegVideoEncoder::close() {
    destroy();
}

REGISTER_VIDEO_ENCODER("ffmpeg_h264", FFmpegVideoEncoder)

} // namespace hal
} // namespace ai_stream
