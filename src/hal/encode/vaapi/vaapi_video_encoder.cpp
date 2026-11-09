// src/hal/encode/vaapi/vaapi_video_encoder.cpp
#include "vaapi_video_encoder.h"
#include "ai_stream/hal/i_video_encoder.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libswscale/swscale.h>
}

namespace ai_stream {
namespace hal {
namespace {

std::string vaapiDevicePath(int device_id) {
    const char* env = std::getenv("AI_STREAM_VAAPI_DEVICE");
    if (env && *env) return env;
    char buf[64];
    // renderD128 对应 device_id=0，依次递增
    std::snprintf(buf, sizeof(buf), "/dev/dri/renderD%d", 128 + (device_id > 0 ? device_id : 0));
    return buf;
}

} // namespace

VaapiVideoEncoder::~VaapiVideoEncoder() {
    destroy();
}

bool VaapiVideoEncoder::isAvailable() const {
    if (av_hwdevice_find_type_by_name("vaapi") == AV_HWDEVICE_TYPE_NONE) {
        return false;
    }
    if (!avcodec_find_encoder_by_name("h264_vaapi")) {
        return false;
    }
    const std::string device = vaapiDevicePath(0);
    return ::access(device.c_str(), R_OK | W_OK) == 0;
}

bool VaapiVideoEncoder::open(const VideoEncoderConfig& config) {
    if (opened_) return true;

    width_ = config.width;
    height_ = config.height;
    if (width_ <= 0 || height_ <= 0) {
        LOG_ERROR("[VaapiVideoEncoder] Invalid frame size");
        return false;
    }

    const std::string device = vaapiDevicePath(config.device_id);
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_VAAPI,
                                     device.c_str(), nullptr, 0);
    if (ret < 0) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[VaapiVideoEncoder] Failed to create VAAPI device '{}': {}",
                      device, errbuf);
        destroy();
        return false;
    }

    const AVCodec* codec = avcodec_find_encoder_by_name("h264_vaapi");
    if (!codec) {
        LOG_ERROR("[VaapiVideoEncoder] h264_vaapi encoder not found in FFmpeg");
        destroy();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        destroy();
        return false;
    }

    codec_ctx_->codec_id = AV_CODEC_ID_H264;
    codec_ctx_->codec_type = AVMEDIA_TYPE_VIDEO;
    codec_ctx_->width = width_;
    codec_ctx_->height = height_;
    codec_ctx_->pix_fmt = AV_PIX_FMT_VAAPI;
    codec_ctx_->time_base = AVRational{1, config.fps > 0 ? config.fps : 25};
    codec_ctx_->framerate = AVRational{config.fps > 0 ? config.fps : 25, 1};
    codec_ctx_->gop_size = config.gop > 0 ? config.gop : (config.fps > 0 ? config.fps : 25);
    // bitrate 未配置时给 4Mbps 兜底，避免 VAAPI CBR 因 0 码率打开失败
    const int bitrate_kbps = config.bitrate_kbps > 0 ? config.bitrate_kbps : 4000;
    codec_ctx_->bit_rate = static_cast<int64_t>(bitrate_kbps) * 1000;
    codec_ctx_->max_b_frames = 0;  // 直播/录制低延迟
    codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;

    // VAAPI 输出面：NV12（iHD 的 8-bit 标准格式）
    hw_frames_ctx_ = av_hwframe_ctx_alloc(hw_device_ctx_);
    if (!hw_frames_ctx_) {
        destroy();
        return false;
    }
    auto* frames = reinterpret_cast<AVHWFramesContext*>(hw_frames_ctx_->data);
    frames->format = AV_PIX_FMT_VAAPI;
    frames->sw_format = AV_PIX_FMT_NV12;
    frames->width = width_;
    frames->height = height_;
    frames->initial_pool_size = 0;  // 由驱动决定
    ret = av_hwframe_ctx_init(hw_frames_ctx_);
    if (ret < 0) {
        LOG_ERROR_FMT("[VaapiVideoEncoder] av_hwframe_ctx_init failed: {}", ret);
        destroy();
        return false;
    }
    codec_ctx_->hw_frames_ctx = av_buffer_ref(hw_frames_ctx_);

    // 低延迟 CBR；部分老驱动不支持这些私有选项，失败不阻断
    av_opt_set(codec_ctx_->priv_data, "rc_mode", "CBR", 0);
    av_opt_set_int(codec_ctx_->priv_data, "async_depth", 1, 0);

    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[VaapiVideoEncoder] avcodec_open2 failed: {}", errbuf);
        destroy();
        return false;
    }

    if (codec_ctx_->extradata_size > 0) {
        extradata_.assign(codec_ctx_->extradata,
                          codec_ctx_->extradata + codec_ctx_->extradata_size);
    } else {
        LOG_WARN("[VaapiVideoEncoder] Encoder returned no extradata (SPS/PPS)");
    }

    sw_frame_ = av_frame_alloc();
    if (!sw_frame_) {
        destroy();
        return false;
    }
    sw_frame_->format = AV_PIX_FMT_NV12;
    sw_frame_->width = width_;
    sw_frame_->height = height_;
    if (av_frame_get_buffer(sw_frame_, 32) < 0) {
        destroy();
        return false;
    }

    hw_frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!hw_frame_ || !packet_) {
        destroy();
        return false;
    }

    sws_ctx_ = sws_getContext(width_, height_, AV_PIX_FMT_YUV420P,
                              width_, height_, AV_PIX_FMT_NV12,
                              SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        destroy();
        return false;
    }

    opened_ = true;
    LOG_INFO_FMT("[VaapiVideoEncoder] Opened: h264_vaapi {}x{} @ {}kbps (device: {})",
                 width_, height_, bitrate_kbps, device);
    return true;
}

bool VaapiVideoEncoder::encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                               std::vector<EncodedPacket>& packets) {
    if (!opened_ || !codec_ctx_) return false;

    const size_t need = static_cast<size_t>(width_) * height_ * 3 / 2;
    if (!yuv420p || size < need) return false;
    if (av_frame_make_writable(sw_frame_) < 0) return false;

    // 上一轮包数据只保证到本次 encode() 之前有效
    staging_frames_.clear();

    // 连续 YUV420P → NV12（三平面指针直接指向输入，无需整帧拷贝）
    const uint8_t* src_y = yuv420p;
    const uint8_t* src_u = yuv420p + static_cast<size_t>(width_) * height_;
    const uint8_t* src_v = src_u + static_cast<size_t>(width_) * height_ / 4;
    const uint8_t* src_data[4] = {src_y, src_u, src_v, nullptr};
    int src_linesize[4] = {width_, width_ / 2, width_ / 2, 0};
    sws_scale(sws_ctx_, src_data, src_linesize, 0, height_,
              sw_frame_->data, sw_frame_->linesize);

    // 上传到 VAAPI 面（每帧重新取 buffer；send_frame 内部持有引用）
    av_frame_unref(hw_frame_);
    hw_frame_->format = AV_PIX_FMT_VAAPI;
    hw_frame_->width = width_;
    hw_frame_->height = height_;
    int ret = av_hwframe_get_buffer(codec_ctx_->hw_frames_ctx, hw_frame_, 0);
    if (ret < 0) {
        LOG_ERROR_FMT("[VaapiVideoEncoder] av_hwframe_get_buffer failed: {}", ret);
        return false;
    }
    ret = av_hwframe_transfer_data(hw_frame_, sw_frame_, 0);
    if (ret < 0) {
        LOG_ERROR_FMT("[VaapiVideoEncoder] upload frame failed: {}", ret);
        return false;
    }
    hw_frame_->pts = pts;

    ret = avcodec_send_frame(codec_ctx_, hw_frame_);
    if (ret < 0) {
        if (ret == AVERROR(EAGAIN)) {
            drainPackets(packets);
            ret = avcodec_send_frame(codec_ctx_, hw_frame_);
        }
        if (ret < 0) {
            LOG_ERROR_FMT("[VaapiVideoEncoder] send_frame failed: {}", ret);
            return false;
        }
    }
    return drainPackets(packets);
}

bool VaapiVideoEncoder::drainPackets(std::vector<EncodedPacket>& packets) {
    for (;;) {
        int ret = avcodec_receive_packet(codec_ctx_, packet_);
        if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
        if (ret < 0) {
            LOG_ERROR_FMT("[VaapiVideoEncoder] receive_packet failed: {}", ret);
            return false;
        }
        staging_frames_.emplace_back(packet_->data, packet_->data + packet_->size);
        const auto& buf = staging_frames_.back();
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

bool VaapiVideoEncoder::flush(std::vector<EncodedPacket>& packets) {
    if (!opened_ || !codec_ctx_) return false;
    staging_frames_.clear();
    int ret = avcodec_send_frame(codec_ctx_, nullptr);
    if (ret < 0 && ret != AVERROR_EOF) {
        LOG_ERROR_FMT("[VaapiVideoEncoder] flush send_frame failed: {}", ret);
        return false;
    }
    return drainPackets(packets);
}

void VaapiVideoEncoder::close() {
    destroy();
}

void VaapiVideoEncoder::destroy() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (packet_) av_packet_free(&packet_);
    if (hw_frame_) av_frame_free(&hw_frame_);
    if (sw_frame_) av_frame_free(&sw_frame_);
    if (codec_ctx_) avcodec_free_context(&codec_ctx_);
    if (hw_frames_ctx_) av_buffer_unref(&hw_frames_ctx_);
    if (hw_device_ctx_) av_buffer_unref(&hw_device_ctx_);
    extradata_.clear();
    staging_frames_.clear();
    opened_ = false;
}

REGISTER_VIDEO_ENCODER("vaapi_h264", VaapiVideoEncoder)

} // namespace hal
} // namespace ai_stream
