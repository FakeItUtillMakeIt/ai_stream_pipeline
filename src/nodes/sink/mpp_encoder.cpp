// src/nodes/sink/mpp_encoder.cpp
// MPP 硬编码适配层实现——见头文件说明
#include "mpp_encoder.h"
#include "ai_stream/hal/i_video_encoder.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <cstring>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

#include <vector>

namespace ai_stream {
namespace nodes {

MppEncoder::~MppEncoder() {
    close();
}

bool MppEncoder::openOutput(const std::string& url, const std::string& format_name) {
    // mp4 → FileEncoder 语义，其余（flv）→ RTMPEncoder 语义
    const char* fmt = (format_name == "mp4") ? "mp4" : "flv";
    int ret = avformat_alloc_output_context2(&fmt_ctx_, nullptr, fmt, url.c_str());
    if (ret < 0 || !fmt_ctx_) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[MppEncoder] Failed to allocate output context: {}", errbuf);
        return false;
    }
    if (!(fmt_ctx_->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&fmt_ctx_->pb, url.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) {
            char errbuf[256] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[MppEncoder] Failed to open output: {}", errbuf);
            avformat_free_context(fmt_ctx_);
            fmt_ctx_ = nullptr;
            return false;
        }
    }
    LOG_INFO_FMT("[MppEncoder] Output opened: {}", url);
    return true;
}

bool MppEncoder::openVideoCodec() {
    // 1. 通过 HAL 工厂创建 MPP 编码器
    hal_encoder_ = hal::VideoEncoderFactory::instance().create("mpp_h264");
    if (!hal_encoder_ || !hal_encoder_->isAvailable()) {
        LOG_WARN("[MppEncoder] mpp_h264 backend unavailable, caller should fall back");
        hal_encoder_.reset();
        return false;
    }

    hal::VideoEncoderConfig cfg;
    cfg.width = width_;
    cfg.height = height_;
    cfg.bitrate_kbps = static_cast<int>(codec_ctx_->bit_rate / 1000);
    cfg.fps = codec_ctx_->time_base.den;  // addVideoStream 设置为 25
    cfg.gop = codec_ctx_->time_base.den;

    if (!hal_encoder_->open(cfg)) {
        LOG_ERROR("[MppEncoder] HAL encoder open failed");
        hal_encoder_.reset();
        return false;
    }

    // 2. codec_ctx 仅作封装载体：不 avcodec_open2，手工填充参数与序列头
    size_t extradata_size = 0;
    const uint8_t* extradata = hal_encoder_->getExtradata(extradata_size);
    if (!extradata || extradata_size == 0) {
        LOG_ERROR("[MppEncoder] HAL encoder produced no extradata");
        return false;
    }
    codec_ctx_->extradata = static_cast<uint8_t*>(av_mallocz(extradata_size + AV_INPUT_BUFFER_PADDING_SIZE));
    memcpy(codec_ctx_->extradata, extradata, extradata_size);
    codec_ctx_->extradata_size = static_cast<int>(extradata_size);

    // 3. codecpar 供容器封装（FLV/MP4 需要 extradata）
    if (avcodec_parameters_from_context(video_stream_->codecpar, codec_ctx_) < 0) {
        LOG_ERROR("[MppEncoder] avcodec_parameters_from_context failed");
        return false;
    }

    LOG_INFO("[MppEncoder] Opened (HAL mpp_h264, AnnexB→AVCC extradata ready)");
    return true;
}

bool MppEncoder::encodeFrame(const uint8_t* data, int width, int height,
                             int step, int64_t pts) {
    if (!hal_encoder_ || !initialized_) return false;
    if (!data || width <= 0 || height <= 0 || step <= 0) return false;

    // BGR → YUV420P（复用基类 sws 与 av_frame）
    if (av_frame_make_writable(av_frame_) < 0) {
        LOG_ERROR("[MppEncoder] av_frame_make_writable failed");
        return false;
    }
    const uint8_t* src_data[1] = {data};
    const int src_linesize[1] = {step};
    sws_scale(sws_ctx_, src_data, src_linesize, 0, height,
              av_frame_->data, av_frame_->linesize);

    // YUV420P 连续化（av_frame 三平面 → 单缓冲；行 stride 修复）
    const int w = av_frame_->width;
    const int h = av_frame_->height;
    std::vector<uint8_t> yuv(static_cast<size_t>(w) * h * 3 / 2);
    uint8_t* dst_y = yuv.data();
    uint8_t* dst_u = dst_y + static_cast<size_t>(w) * h;
    uint8_t* dst_v = dst_u + static_cast<size_t>(w) * h / 4;
    for (int y0 = 0; y0 < h; ++y0) {
        memcpy(dst_y + static_cast<size_t>(y0) * w,
               av_frame_->data[0] + static_cast<size_t>(y0) * av_frame_->linesize[0], w);
    }
    for (int y0 = 0; y0 < h / 2; ++y0) {
        memcpy(dst_u + static_cast<size_t>(y0) * (w / 2),
               av_frame_->data[1] + static_cast<size_t>(y0) * av_frame_->linesize[1], w / 2);
        memcpy(dst_v + static_cast<size_t>(y0) * (w / 2),
               av_frame_->data[2] + static_cast<size_t>(y0) * av_frame_->linesize[2], w / 2);
    }

    // HAL 编码（pts 由 sink 节点传入，保持音画时间基一致）
    std::vector<hal::EncodedPacket> packets;
    if (!hal_encoder_->encode(yuv.data(), yuv.size(), pts, packets)) {
        LOG_ERROR("[MppEncoder] HAL encode failed");
        return false;
    }

    // 写容器
    for (const auto& p : packets) {
        AVPacket* pkt = av_packet_alloc();
        if (!pkt) return false;
        // 数据生命周期仅本函数内，拷贝进 AVPacket
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
            LOG_ERROR_FMT("[MppEncoder] write frame failed: {}", errbuf);
            return false;
        }
    }
    return true;
}

void MppEncoder::flush() {
    // 硬编码器无需 drain：直接写文件尾
    if (fmt_ctx_ && fmt_ctx_->pb) {
        av_write_trailer(fmt_ctx_);
    }
}

void MppEncoder::close() {
    if (hal_encoder_) {
        hal_encoder_->close();
        hal_encoder_.reset();
    }
}

} // namespace nodes
} // namespace ai_stream
