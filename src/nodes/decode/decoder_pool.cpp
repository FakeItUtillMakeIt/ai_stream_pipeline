// src/nodes/decode/decoder_pool.cpp
#include "decoder_pool.h"
#include "3rd_party/log_mgr/log_mgr.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace ai_stream {
namespace nodes {

DecoderContext::~DecoderContext() {
    if (d_bgr_buffer){
#ifdef WITH_CUDA
        cudaFree(d_bgr_buffer);
#endif
    }
    if (hw_frame) {
        av_frame_free(&hw_frame);
    }
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
    }
    if (bgr_buffer) {
        av_free(bgr_buffer);
        bgr_buffer = nullptr;
    }
    if (bgr_frame) {
        av_frame_free(&bgr_frame);
    }
    if (frame) {
        av_frame_free(&frame);
    }
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
        sws_ctx = nullptr;
    }
    if (codec_ctx) {
        avcodec_free_context(&codec_ctx);
    }
}

DecoderPool::DecoderPool() {
    LOG_DEBUG_FMT("[DecoderPool] Created");
}

DecoderPool::~DecoderPool() {
    clear();
    LOG_DEBUG_FMT("[DecoderPool] Destroyed");
}

std::shared_ptr<DecoderContext> DecoderPool::getDecoder(uint32_t stream_id, int codec_id,
                                                          bool use_hw,
                                                          const uint8_t* extradata,
                                                          int extradata_size) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = decoders_.find(stream_id);
    if (it != decoders_.end()) {
        return it->second;
    }

    // 创建新的解码器
    auto ctx = createDecoder(codec_id, use_hw, extradata, extradata_size);
    if (ctx) {
        decoders_[stream_id] = ctx;
        LOG_INFO_FMT("[DecoderPool] Created decoder for stream_id={}, codec_id={}", stream_id, codec_id);
    }
    return ctx;
}

void DecoderPool::releaseDecoder(uint32_t stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = decoders_.find(stream_id);
    if (it != decoders_.end()) {
        LOG_INFO_FMT("[DecoderPool] Released decoder for stream_id={}", stream_id);
        decoders_.erase(it);
    }
}

size_t DecoderPool::getActiveCount() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return decoders_.size();
}

void DecoderPool::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    decoders_.clear();
}

std::shared_ptr<DecoderContext> DecoderPool::createDecoder(int codec_id, bool use_hw,
                                                              const uint8_t* extradata,
                                                              int extradata_size) {
    auto ctx = std::make_shared<DecoderContext>();
    ctx->codec_id = codec_id;
    ctx->use_hw = use_hw;
    
    const AVCodec* codec = nullptr;
    
    if (use_hw) {
        // 优先使用 cuvid 专用解码器
        if (codec_id == AV_CODEC_ID_H264) {
            codec = avcodec_find_decoder_by_name("h264_cuvid");
        } else if (codec_id == AV_CODEC_ID_HEVC) {
            codec = avcodec_find_decoder_by_name("hevc_cuvid");
        }
        
        if (!codec) {
            LOG_WARN_FMT("[DecoderPool] HW decoder not found for codec_id={}, fallback to SW", codec_id);
            ctx->use_hw = false;
        }
    }
    
    if (!ctx->use_hw) {
        codec = avcodec_find_decoder((AVCodecID)codec_id);
    }
    
    if (!codec) {
        LOG_ERROR_FMT("[DecoderPool] Failed to find decoder for codec_id={}", codec_id);
        return nullptr;
    }
    
    // 分配解码器上下文
    ctx->codec_ctx = avcodec_alloc_context3(codec);
    if (!ctx->codec_ctx) {
        LOG_ERROR_FMT("[DecoderPool] Failed to allocate codec context");
        return nullptr;
    }
    
    if (extradata && extradata_size > 0) {
        ctx->codec_ctx->extradata = (uint8_t*)av_mallocz(extradata_size + AV_INPUT_BUFFER_PADDING_SIZE);
        if (ctx->codec_ctx->extradata) {
            memcpy(ctx->codec_ctx->extradata, extradata, extradata_size);
            ctx->codec_ctx->extradata_size = extradata_size;
            LOG_INFO_FMT("[DecoderPool] Set extradata: {} bytes", extradata_size);
        }
    }

    if (ctx->use_hw) {
#ifdef WITH_CUDA
        // 初始化 CUDA 设备上下文
        int ret = av_hwdevice_ctx_create(&ctx->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA,
                                          nullptr, nullptr, 0);
        if (ret < 0) {
            char errbuf[256];
            av_strerror(ret, errbuf, sizeof(errbuf));
            LOG_WARN_FMT("[DecoderPool] av_hwdevice_ctx_create failed: {}, fallback to SW", errbuf);
            ctx->use_hw = false;
            avcodec_free_context(&ctx->codec_ctx);
            return createDecoder(codec_id, false);  // 递归 fallback
        }

        ctx->codec_ctx->hw_device_ctx = av_buffer_ref(ctx->hw_device_ctx);
        ctx->codec_ctx->thread_count = 1;  // 硬件解码不需要多线程
#else
        LOG_WARN("[DecoderPool] HW decode requested but CUDA not available, fallback to SW");
        ctx->use_hw = false;
#endif
    } else {
        ctx->codec_ctx->thread_count = 4;
        ctx->codec_ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;
    }
    
    if (avcodec_open2(ctx->codec_ctx, codec, nullptr) < 0) {
        LOG_ERROR_FMT("[DecoderPool] Failed to open codec");
        return nullptr;
    }
    
    ctx->frame = av_frame_alloc();
    ctx->bgr_frame = av_frame_alloc();
    if (ctx->use_hw) {
        ctx->hw_frame = av_frame_alloc();
    }
    if(!ctx->frame || !ctx->bgr_frame || (ctx->use_hw && !ctx->hw_frame))
    {
        LOG_ERROR_FMT("[DecoderPool] Failed to allocate frames");
        return nullptr;
    }
    ctx->initialized = true;
    LOG_INFO_FMT("[DecoderPool] Decoder created: {} (hw={})", codec->name, ctx->use_hw);
    return ctx;
}


} // namespace nodes
} // namespace ai_stream