// src/hal/nvdec/nvdec_video_codec.cpp
// NVIDIA NVDEC 硬件视频解码——封装 FFmpeg + CUDA hwaccel 到 HAL 接口
#include "nvdec_video_codec.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <cuda_runtime.h>

extern "C" {
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libavutil/hwcontext_cuda.h>
}

namespace ai_stream {
namespace hal {

NvdecVideoCodec::NvdecVideoCodec() {
    LOG_DEBUG("[NvdecVideoCodec] Constructor");
}

NvdecVideoCodec::~NvdecVideoCodec() {
    cleanup();
    LOG_DEBUG("[NvdecVideoCodec] Destroyed");
}

bool NvdecVideoCodec::isAvailable() const {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return false;
    }

    // 检查 FFmpeg 是否支持 CUDA hwaccel
    AVHWDeviceType hw_type = av_hwdevice_find_type_by_name("cuda");
    if (hw_type == AV_HWDEVICE_TYPE_NONE) {
        return false;
    }

    // 检查是否有支持的解码器
    const AVCodec* codec = nullptr;
    void* opaque = nullptr;
    while ((codec = av_codec_iterate(&opaque))) {
        if (av_codec_is_decoder(codec) && codec->type == AVMEDIA_TYPE_VIDEO) {
            // 检查此编解码器是否支持 CUDA hwaccel
            for (int i = 0;; i++) {
                const AVCodecHWConfig* config = avcodec_get_hw_config(codec, i);
                if (!config) break;
                if (config->device_type == AV_HWDEVICE_TYPE_CUDA) {
                    return true;
                }
            }
        }
    }

    return false;
}

bool NvdecVideoCodec::init(const std::string& codec_name,
                            const uint8_t* extradata,
                            int extradata_size) {
    codec_name_ = codec_name;
    extradata_ = extradata;
    extradata_size_ = extradata_size;

    LOG_INFO_FMT("[NvdecVideoCodec] Initializing codec: {}", codec_name);
    initialized_ = initDecoder();
    return initialized_;
}

bool NvdecVideoCodec::initDecoder() {
    // 查找解码器 - 优先使用 cuvid，否则使用标准解码器 + hwaccel
    AVCodecID codec_id = AV_CODEC_ID_NONE;
    const char* cuvid_name = nullptr;

    if (codec_name_ == "h264" || codec_name_ == "H264") {
        codec_id = AV_CODEC_ID_H264;
        cuvid_name = "h264_cuvid";
    } else if (codec_name_ == "h265" || codec_name_ == "hevc" || codec_name_ == "H265") {
        codec_id = AV_CODEC_ID_HEVC;
        cuvid_name = "hevc_cuvid";
    } else if (codec_name_ == "vp9" || codec_name_ == "VP9") {
        codec_id = AV_CODEC_ID_VP9;
    } else if (codec_name_ == "av1" || codec_name_ == "AV1") {
        codec_id = AV_CODEC_ID_AV1;
    }

    const AVCodec* codec = nullptr;

    // 首先尝试 cuvid 专用解码器
    if (cuvid_name) {
        codec = avcodec_find_decoder_by_name(cuvid_name);
        if (codec) {
            LOG_INFO_FMT("[NvdecVideoCodec] Found cuvid decoder: {}", cuvid_name);
        }
    }

    // 回退到标准解码器
    if (!codec && codec_id != AV_CODEC_ID_NONE) {
        codec = avcodec_find_decoder(codec_id);
    }

    // 最后尝试通过名称查找
    if (!codec) {
        codec = avcodec_find_decoder_by_name(codec_name_.c_str());
    }

    if (!codec) {
        LOG_ERROR_FMT("[NvdecVideoCodec] Codec not found: {}", codec_name_);
        return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
        LOG_ERROR("[NvdecVideoCodec] Failed to allocate codec context");
        return false;
    }

    // 尝试设置 CUDA hwaccel
    AVHWDeviceType hw_type = av_hwdevice_find_type_by_name("cuda");
    if (hw_type != AV_HWDEVICE_TYPE_NONE) {
        int ret = av_hwdevice_ctx_create(&hw_device_ctx_, hw_type, nullptr, nullptr, 0);
        if (ret == 0) {
            codec_ctx_->hw_device_ctx = av_buffer_ref(hw_device_ctx_);

            // 设置 get_format 回调以使用 CUDA 格式
            codec_ctx_->get_format = [](AVCodecContext* ctx, const enum AVPixelFormat* pix_fmts) -> enum AVPixelFormat {
                for (const enum AVPixelFormat* p = pix_fmts; *p != AV_PIX_FMT_NONE; p++) {
                    if (*p == AV_PIX_FMT_CUDA) {
                        return *p;
                    }
                }
                // 没有 CUDA 格式可用，使用第一个
                return pix_fmts[0];
            };

            LOG_INFO("[NvdecVideoCodec] CUDA hwaccel enabled");
        } else {
            LOG_WARN("[NvdecVideoCodec] Failed to create CUDA hw device, using software");
            if (hw_device_ctx_) {
                av_buffer_unref(&hw_device_ctx_);
                hw_device_ctx_ = nullptr;
            }
        }
    }

    if (extradata_ && extradata_size_ > 0) {
        codec_ctx_->extradata = static_cast<uint8_t*>(av_malloc(extradata_size_ + AV_INPUT_BUFFER_PADDING_SIZE));
        std::memcpy(codec_ctx_->extradata, extradata_, extradata_size_);
        codec_ctx_->extradata_size = extradata_size_;
    }

    codec_ctx_->thread_count = 1;

    AVDictionary* opts = nullptr;
    int ret = avcodec_open2(codec_ctx_, codec, &opts);
    av_dict_free(&opts);
    if (ret < 0) {
        char errbuf[256];
        av_strerror(ret, errbuf, sizeof(errbuf));
        LOG_ERROR_FMT("[NvdecVideoCodec] Failed to open codec: {} ({})", errbuf, ret);
        return false;
    }

    frame_ = av_frame_alloc();
    hw_frame_ = av_frame_alloc();
    packet_ = av_packet_alloc();

    if (!frame_ || !hw_frame_ || !packet_) {
        LOG_ERROR("[NvdecVideoCodec] Failed to allocate frames/packet");
        return false;
    }

    LOG_INFO_FMT("[NvdecVideoCodec] Decoder initialized: {}", codec->name);
    return true;
}

bool NvdecVideoCodec::decode(const uint8_t* packet_data, int packet_size,
                              DecodedFrame& frame) {
    if (!initialized_ || !codec_ctx_) {
        LOG_ERROR("[NvdecVideoCodec] Not initialized");
        return false;
    }

    packet_->data = const_cast<uint8_t*>(packet_data);
    packet_->size = packet_size;

    int ret = avcodec_send_packet(codec_ctx_, packet_);
    if (ret < 0) {
        if (ret == AVERROR_INVALIDDATA) {
            LOG_WARN("[NvdecVideoCodec] Invalid data, skipping packet");
            return false;
        }
        if (ret == AVERROR(EAGAIN)) {
            return false;
        }
        LOG_ERROR_FMT("[NvdecVideoCodec] avcodec_send_packet failed: {}", ret);
        return false;
    }

    ret = avcodec_receive_frame(codec_ctx_, frame_);
    if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
        return false;
    } else if (ret < 0) {
        LOG_ERROR_FMT("[NvdecVideoCodec] avcodec_receive_frame failed: {}", ret);
        return false;
    }

    // 如果帧在 GPU 内存中，转移到系统内存
    if (frame_->format == AV_PIX_FMT_CUDA && hw_device_ctx_) {
        ret = av_hwframe_transfer_data(hw_frame_, frame_, 0);
        if (ret < 0) {
            LOG_ERROR_FMT("[NvdecVideoCodec] av_hwframe_transfer_data failed: {}", ret);
            return false;
        }
        av_frame_unref(frame_);
        av_frame_move_ref(frame_, hw_frame_);
    }

    frame.data = frame_->data[0];
    frame.width = frame_->width;
    frame.height = frame_->height;
    frame.pitch = frame_->linesize[0];
    frame.format = frame_->format;
    frame.owns_data = false;

    LOG_DEBUG_FMT("[NvdecVideoCodec] Frame decoded: {}x{}, format={}, pitch={}",
                  frame.width, frame.height, frame.format, frame.pitch);

    return true;
}

void NvdecVideoCodec::release() {
    cleanup();
    LOG_DEBUG("[NvdecVideoCodec] Released");
}

void NvdecVideoCodec::cleanup() {
    if (packet_) {
        av_packet_free(&packet_);
    }
    if (hw_frame_) {
        av_frame_free(&hw_frame_);
    }
    if (frame_) {
        av_frame_free(&frame_);
    }
    if (codec_ctx_) {
        avcodec_free_context(&codec_ctx_);
    }
    if (hw_device_ctx_) {
        av_buffer_unref(&hw_device_ctx_);
    }
    initialized_ = false;
}

// 注册 NVDEC 后端到工厂
#ifdef WITH_CUDA
REGISTER_VIDEO_CODEC(VideoCodecBackend::NVDEC, NvdecVideoCodec)
#endif

} // namespace hal
} // namespace ai_stream
