// src/nodes/sink/encoder_base.cpp
#include "encoder_base.h"
#include "3rd_party/log_mgr/log_mgr.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
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
    
    LOG_INFO_FMT("[EncoderBase] Initializing: url={}, format={}, {}x{}, {}kbps",
                 output_url, format_name, width, height, bitrate);
    
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
    
    // 分配帧
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
    
    // 初始化转换器 BGR24 -> YUV420P
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
    LOG_INFO_FMT("[EncoderBase] Initialized successfully");
    return true;
}

bool EncoderBase::addVideoStream(int width, int height, int bitrate,
                                  const std::string& encoder_name) {
    // 查找编码器
    const AVCodec* codec = avcodec_find_encoder_by_name(encoder_name.c_str());
    if (!codec) {
        LOG_WARN_FMT("[EncoderBase] Encoder '{}' not found, trying H264", encoder_name);
        codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    }
    if (!codec) {
        LOG_ERROR("[EncoderBase] No encoder found");
        return false;
    }
    
    LOG_INFO_FMT("[EncoderBase] Using encoder: {}", codec->name);
    
    // 创建流
    video_stream_ = avformat_new_stream(fmt_ctx_, codec);
    if (!video_stream_) {
        LOG_ERROR("[EncoderBase] Failed to create stream");
        return false;
    }
    video_stream_->id = fmt_ctx_->nb_streams - 1;
    
    // 分配编码器上下文
    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        LOG_ERROR("[EncoderBase] Failed to allocate codec context");
        return false;
    }
    
    // 设置编码参数
    codec_ctx_->codec_id = codec->id;
    codec_ctx_->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx_->bit_rate = bitrate * 1000;
    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = (AVRational){1, 25};
    codec_ctx_->framerate = (AVRational){25, 1};
    codec_ctx_->gop_size = 12;
    codec_ctx_->max_b_frames = 2;
    codec_ctx_->pix_fmt = AV_PIX_FMT_YUV420P;
    
    // 设置全局头标志
    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
        codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }
    
    video_stream_->time_base = codec_ctx_->time_base;
    
    // 设置 H.264 特定选项
    if (codec_ctx_->codec_id == AV_CODEC_ID_H264) {
        av_opt_set(codec_ctx_->priv_data, "preset", "fast", 0);
        av_opt_set(codec_ctx_->priv_data, "tune", "zerolatency", 0);
        av_opt_set(codec_ctx_->priv_data, "crf", "23", 0);
    }
    
    LOG_INFO_FMT("[EncoderBase] Video stream added: {}x{} @ {}kbps", width, height, bitrate);
    return true;
}

bool EncoderBase::openVideoCodec() {
    const AVCodec* codec = avcodec_find_encoder(codec_ctx_->codec_id);
    if (!codec) {
        LOG_ERROR("[EncoderBase] Codec not found");
        return false;
    }
    
    int ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[EncoderBase] Failed to open codec: {}", errbuf);
        return false;
    }
    
    ret = avcodec_parameters_from_context(video_stream_->codecpar, codec_ctx_);
    if (ret < 0) {
        LOG_ERROR("[EncoderBase] Failed to copy codec parameters");
        return false;
    }
    
    LOG_INFO("[EncoderBase] Video codec opened");
    return true;
}

bool EncoderBase::writeHeader() {
    if (!fmt_ctx_) {
        LOG_ERROR("[EncoderBase] Format context is null");
        return false;
    }
    
    // 打印格式信息
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
    if (!data)
    {
        LOG_ERROR("[EncoderBase] Invalid input data");
        return false;
    }
    if (width <= 0 || height <= 0 || step <= 0)
    {
        LOG_ERROR("[EncoderBase] Invalid input size");
        return false;
    }
    if (!initialized_) {
        LOG_ERROR("[EncoderBase] Not initialized");
        return false;
    }
    
    // 确保帧可写
    int ret = av_frame_make_writable(av_frame_);
    if (ret < 0) {
        LOG_ERROR("[EncoderBase] Failed to make frame writable");
        return false;
    }
    
    // BGR24 -> YUV420P 转换
    const uint8_t* src_data[1] = { data };
    int src_linesize[1] = { step };
    
    sws_scale(sws_ctx_, src_data, src_linesize, 0, height,
              av_frame_->data, av_frame_->linesize);
    
    av_frame_->pts = pts;
    
    // 发送帧到编码器
    ret = avcodec_send_frame(codec_ctx_, av_frame_);
    if (ret < 0) {
        if (ret == AVERROR(EAGAIN)) {
            // 编码器需要先读取输出
        } else {
            char errbuf[256] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[EncoderBase] Failed to send frame: {}", errbuf);
            return false;
        }
    }
    
    // 接收编码后的包
    AVPacket* pkt = av_packet_alloc();
    if (!pkt) {
        LOG_ERROR("[EncoderBase] Failed to allocate packet");
        return false;
    }
    
    while (ret >= 0) {
        ret = avcodec_receive_packet(codec_ctx_, pkt);
        if (ret == AVERROR(EAGAIN)) {
            break;
        }
        if (ret == AVERROR_EOF) {
            break;
        }
        if (ret < 0) {
            char errbuf[256] = {0};
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_ERROR_FMT("[EncoderBase] Failed to receive packet: {}", errbuf);
            av_packet_free(&pkt);
            return false;
        }
        
        // 设置时间基
        av_packet_rescale_ts(pkt, codec_ctx_->time_base, video_stream_->time_base);
        pkt->stream_index = video_stream_->index;
        
        // 写入输出
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

void EncoderBase::flush() {
    if (!initialized_ || !fmt_ctx_ ||!codec_ctx_ || closed_.load()) return;
    
    LOG_INFO("[EncoderBase] Flushing encoder...");
    
    // 发送空帧冲刷编码器
    int ret=avcodec_send_frame(codec_ctx_, nullptr);
    if (ret<0 && ret!=AVERROR_EOF)
    {
        LOG_WARN_FMT("[EncoderBase] Failed to send frame: {}", ret);
    }
    
    // 接收剩余的包
    AVPacket* pkt = av_packet_alloc();
    if (pkt) {
        while (true) {
            ret = avcodec_receive_packet(codec_ctx_, pkt);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) break;
            
            av_packet_rescale_ts(pkt, codec_ctx_->time_base, video_stream_->time_base);
            pkt->stream_index = video_stream_->index;
            av_interleaved_write_frame(fmt_ctx_, pkt);
            av_packet_unref(pkt);
        }
        av_packet_free(&pkt);
    }
    
    // 写文件尾
    if (fmt_ctx_ && fmt_ctx_->pb) {
        ret = av_write_trailer(fmt_ctx_);
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
    if(closed_.exchange(true))return;

    if (initialized_) {
        flush();
        initialized_ = false;
    }

    //确保所有编码器输出都被读取
    if (codec_ctx_)
    {
        int ret;
        AVPacket* pkt = av_packet_alloc();
        if(pkt)
        {
            while ((ret = avcodec_receive_packet(codec_ctx_,pkt)) >= 0)
            {
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
    
    // 打开 IO
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