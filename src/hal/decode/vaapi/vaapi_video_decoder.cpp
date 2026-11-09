// src/hal/decode/vaapi/vaapi_video_decoder.cpp
#include "vaapi_video_decoder.h"
#include "ai_stream/hal/video_decoder_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unistd.h>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace ai_stream {
namespace hal {
namespace {

constexpr const char* kDefaultVaapiDevice = "/dev/dri/renderD128";

std::string vaapiDevicePath() {
    const char* env = std::getenv("AI_STREAM_VAAPI_DEVICE");
    if (env && *env) return env;
    return kDefaultVaapiDevice;
}

bool vaapiDeviceAccessible() {
    return ::access(vaapiDevicePath().c_str(), R_OK | W_OK) == 0;
}

// 解码器只会选择 VAAPI 像素格式；若不支持则返回 NONE 让 FFmpeg 报错，
// 由节点在 AUTO 模式下回退软件解码。
enum AVPixelFormat vaapiGetFormat(AVCodecContext* /*ctx*/,
                                  const enum AVPixelFormat* pix_fmts) {
    for (const enum AVPixelFormat* p = pix_fmts; p && *p != AV_PIX_FMT_NONE; ++p) {
        if (*p == AV_PIX_FMT_VAAPI) return *p;
    }
    return AV_PIX_FMT_NONE;
}

AVCodecID codecIdFromName(const std::string& name) {
    if (name == "h264" || name == "H264") return AV_CODEC_ID_H264;
    if (name == "h265" || name == "hevc" || name == "H265") return AV_CODEC_ID_HEVC;
    if (name == "vp9" || name == "VP9") return AV_CODEC_ID_VP9;
    if (name == "av1" || name == "AV1") return AV_CODEC_ID_AV1;
    return AV_CODEC_ID_NONE;
}

bool decoderSupportsVaapi(AVCodecID id) {
    const AVCodec* codec = avcodec_find_decoder(id);
    if (!codec) return false;
    for (int i = 0;; ++i) {
        const AVCodecHWConfig* cfg = avcodec_get_hw_config(codec, i);
        if (!cfg) break;
        if (cfg->device_type == AV_HWDEVICE_TYPE_VAAPI &&
            (cfg->methods & AV_CODEC_HW_CONFIG_METHOD_HW_DEVICE_CTX)) {
            return true;
        }
    }
    return false;
}

} // namespace

VaapiVideoDecoder::~VaapiVideoDecoder() {
    cleanup();
}

bool VaapiVideoDecoder::isAvailable() const {
    // 1) FFmpeg 本身识得 VAAPI（Ubuntu 官方包默认启用）
    if (av_hwdevice_find_type_by_name("vaapi") == AV_HWDEVICE_TYPE_NONE) {
        return false;
    }
    // 2) 渲染节点存在且当前用户可读写（需要 render/video 组权限）
    if (!vaapiDeviceAccessible()) {
        static std::once_flag warned;
        std::call_once(warned, [] {
            LOG_WARN_FMT("[VaapiVideoDecoder] VAAPI device '{}' is not accessible "
                         "(add the service user to the 'render' group: "
                         "sudo usermod -aG render $USER); falling back to software decode",
                         vaapiDevicePath());
        });
        return false;
    }
    // 3) 至少 H.264 解码器支持 VAAPI device context
    return decoderSupportsVaapi(AV_CODEC_ID_H264);
}

bool VaapiVideoDecoder::init(const std::string& codec_name,
                             const uint8_t* extradata,
                             int extradata_size) {
    codec_name_ = codec_name;
    extradata_ = extradata;
    extradata_size_ = extradata_size;

    LOG_INFO_FMT("[VaapiVideoDecoder] Initializing codec: {} (device: {})",
                 codec_name, vaapiDevicePath());
    initialized_ = initDecoder();
    return initialized_;
}

bool VaapiVideoDecoder::initDecoder() {
    AVCodecID codec_id = codecIdFromName(codec_name_);
    const AVCodec* codec = codec_id != AV_CODEC_ID_NONE
                               ? avcodec_find_decoder(codec_id)
                               : avcodec_find_decoder_by_name(codec_name_.c_str());
    if (!codec) {
        LOG_ERROR_FMT("[VaapiVideoDecoder] Codec not found: {}", codec_name_);
        return false;
    }

    hw_device_ctx_ = nullptr;
    int ret = av_hwdevice_ctx_create(&hw_device_ctx_, AV_HWDEVICE_TYPE_VAAPI,
                                     vaapiDevicePath().c_str(), nullptr, 0);
    if (ret < 0) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[VaapiVideoDecoder] Failed to create VAAPI device '{}': {}",
                      vaapiDevicePath(), errbuf);
        cleanup();
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        cleanup();
        return false;
    }

    if (extradata_ && extradata_size_ > 0) {
        codec_ctx_->extradata = static_cast<uint8_t*>(
            av_mallocz(extradata_size_ + AV_INPUT_BUFFER_PADDING_SIZE));
        if (!codec_ctx_->extradata) {
            cleanup();
            return false;
        }
        std::memcpy(codec_ctx_->extradata, extradata_, extradata_size_);
        codec_ctx_->extradata_size = extradata_size_;
    }

    codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);
    codec_ctx_->get_format = vaapiGetFormat;
    // 低延迟：解码器不额外缓存输出帧
    codec_ctx_->flags |= AV_CODEC_FLAG_LOW_DELAY;

    ret = avcodec_open2(codec_ctx_, codec, nullptr);
    if (ret < 0) {
        char errbuf[256] = {0};
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[VaapiVideoDecoder] avcodec_open2 failed: {}", errbuf);
        cleanup();
        return false;
    }

    frame_ = av_frame_alloc();
    sw_frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();
    if (!frame_ || !sw_frame_ || !packet_) {
        cleanup();
        return false;
    }

    LOG_INFO_FMT("[VaapiVideoDecoder] Opened: {} ({})",
                 codec->name, vaapiDevicePath());
    return true;
}

bool VaapiVideoDecoder::ensureNv12Converter(int width, int height, int src_format) {
    if (src_format == AV_PIX_FMT_NV12) return true;

    if (nv12_frame_ && sws_ctx_ &&
        sws_width_ == width && sws_height_ == height &&
        sws_src_format_ == src_format) {
        return true;
    }

    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (nv12_frame_) {
        av_frame_free(&nv12_frame_);
    }

    nv12_frame_ = av_frame_alloc();
    if (!nv12_frame_) return false;
    nv12_frame_->format = AV_PIX_FMT_NV12;
    nv12_frame_->width = width;
    nv12_frame_->height = height;
    if (av_frame_get_buffer(nv12_frame_, 32) < 0) {
        av_frame_free(&nv12_frame_);
        return false;
    }

    sws_ctx_ = sws_getContext(width, height, static_cast<AVPixelFormat>(src_format),
                              width, height, AV_PIX_FMT_NV12,
                              SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
    if (!sws_ctx_) {
        av_frame_free(&nv12_frame_);
        return false;
    }
    sws_width_ = width;
    sws_height_ = height;
    sws_src_format_ = src_format;
    LOG_INFO_FMT("[VaapiVideoDecoder] Created {} -> NV12 converter for {}x{}",
                 src_format, width, height);
    return true;
}

bool VaapiVideoDecoder::decode(const uint8_t* packet_data, int packet_size,
                               DecodedFrame& frame) {
    if (!initialized_ || !codec_ctx_) {
        LOG_ERROR("[VaapiVideoDecoder] Not initialized");
        return false;
    }

    av_packet_unref(packet_);
    if (av_new_packet(packet_, packet_size) < 0) {
        LOG_ERROR("[VaapiVideoDecoder] Failed to allocate packet");
        return false;
    }
    std::memcpy(packet_->data, packet_data, packet_size);

    int ret = avcodec_send_packet(codec_ctx_, packet_);
    av_packet_unref(packet_);
    if (ret < 0) {
        LOG_ERROR_FMT("[VaapiVideoDecoder] avcodec_send_packet failed: {}", ret);
        return false;
    }

    av_frame_unref(frame_);
    ret = avcodec_receive_frame(codec_ctx_, frame_);
    if (ret < 0) {
        // EAGAIN：解码器需要更多输入（B 帧重排），本接口一包一帧，
        // 与 FFmpeg 软解后端保持同样的行为。
        if (ret != AVERROR(EAGAIN)) {
            LOG_ERROR_FMT("[VaapiVideoDecoder] avcodec_receive_frame failed: {}", ret);
        }
        return false;
    }

    const AVFrame* out = frame_;
    if (frame_->format == AV_PIX_FMT_VAAPI) {
        av_frame_unref(sw_frame_);
        ret = av_hwframe_transfer_data(sw_frame_, frame_, 0);
        if (ret < 0) {
            LOG_ERROR_FMT("[VaapiVideoDecoder] av_hwframe_transfer_data failed: {}", ret);
            return false;
        }
        out = sw_frame_;
    }

    const int width = out->width;
    const int height = out->height;
    if (width <= 0 || height <= 0) {
        LOG_ERROR("[VaapiVideoDecoder] Invalid decoded frame size");
        return false;
    }

    if (out->format != AV_PIX_FMT_NV12) {
        if (!ensureNv12Converter(width, height, out->format)) {
            LOG_ERROR_FMT("[VaapiVideoDecoder] Unsupported pixel format: {}", out->format);
            return false;
        }
        sws_scale(sws_ctx_, out->data, out->linesize, 0, height,
                  nv12_frame_->data, nv12_frame_->linesize);
        out = nv12_frame_;
    }

    frame.data = out->data[0];
    frame.width = width;
    frame.height = height;
    frame.pitch = out->linesize[0];
    frame.format = AV_PIX_FMT_NV12;
    frame.owns_data = false;  // 借用至下次 decode()
    frame.data_uv = out->data[1];
    frame.pitch_uv = out->linesize[1];
    frame.is_gpu = false;
    frame.d_data = nullptr;
    frame.d_data_uv = nullptr;
    return true;
}

void VaapiVideoDecoder::release() {
    cleanup();
    LOG_DEBUG("[VaapiVideoDecoder] Released");
}

void VaapiVideoDecoder::cleanup() {
    if (sws_ctx_) {
        sws_freeContext(sws_ctx_);
        sws_ctx_ = nullptr;
    }
    if (nv12_frame_) av_frame_free(&nv12_frame_);
    if (sw_frame_) av_frame_free(&sw_frame_);
    if (frame_) av_frame_free(&frame_);
    if (packet_) av_packet_free(&packet_);
    if (codec_ctx_) avcodec_free_context(&codec_ctx_);
    if (hw_device_ctx_) av_buffer_unref(&hw_device_ctx_);
    sws_width_ = sws_height_ = 0;
    sws_src_format_ = -1;
    initialized_ = false;
}

// 注册 VAAPI 后端（编译期始终注册；运行期按设备/驱动可用性选择）
REGISTER_VIDEO_DECODER(VideoDecoderBackend::VAAPI, VaapiVideoDecoder)

} // namespace hal
} // namespace ai_stream
