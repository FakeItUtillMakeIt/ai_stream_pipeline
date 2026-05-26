// src/nodes/decode/decoder_pool.cpp
#include "decoder_pool.h"
#include "3rd_party/log_mgr/log_mgr.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libavutil/hwcontext.h>
#include <libswscale/swscale.h>
}

#include <cuda_runtime.h>

namespace ai_stream {
namespace nodes {

DecoderContext::~DecoderContext() {
    // 释放硬件帧
    if (hw_frame) {
        av_frame_free(&hw_frame);
    }

    // 释放硬件设备上下文
    if (hw_device_ctx) {
        av_buffer_unref(&hw_device_ctx);
    }

    // 释放BGR缓冲区
    if (bgr_buffer) {
        av_free(bgr_buffer);
        bgr_buffer = nullptr;
    }

    // 释放BGR帧
    if (bgr_frame) {
        av_frame_free(&bgr_frame);
    }

    // 释放软件帧
    if (frame) {
        av_frame_free(&frame);
    }

    // 释放格式转换器
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
        sws_ctx = nullptr;
    }

    // 释放解码器上下文（自动清理硬件相关资源）
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

std::shared_ptr<DecoderContext> DecoderPool::getDecoder(uint32_t stream_id, int codec_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // 检查GPU内存，如果不足则清理
    size_t free_mem, total_mem;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
        if (free_mem < 100 * 1024 * 1024) { // 小于100MB时清理
            LOG_WARN_FMT("[DecoderPool] Low GPU memory: {}MB free, clearing all decoders",
                         free_mem / 1024 / 1024);
            decoders_.clear();
        }
    }

    auto it = decoders_.find(stream_id);
    if (it != decoders_.end()) {
        return it->second;
    }
    
    // 创建新的解码器
    auto ctx = createDecoder(codec_id);
    if (ctx) {
        decoders_[stream_id] = ctx;
        LOG_INFO_FMT("[DecoderPool] Created decoder for stream_id={}, codec_id={}", stream_id, codec_id);

        // 记录GPU内存使用
        size_t free_mem, total_mem;
        if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
            LOG_INFO_FMT("[DecoderPool] GPU memory after creating decoder: {}MB free / {}MB total",
                         free_mem / 1024 / 1024, total_mem / 1024 / 1024);
        }
    }
    return ctx;
}

void DecoderPool::releaseDecoder(uint32_t stream_id) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = decoders_.find(stream_id);
    if (it != decoders_.end()) {
        // 确保解码器不在使用中
        if (it->second->in_use.load()) {
            LOG_WARN_FMT("[DecoderPool] Decoder is in use, waiting for release");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // 清理硬件帧和GPU资源
        if (it->second->hw_frame) {
            av_frame_unref(it->second->hw_frame);
        }
        if (it->second->hw_device_ctx) {
            av_buffer_unref(&it->second->hw_device_ctx);
        }

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

void DecoderPool::destroyDecoder(std::shared_ptr<DecoderContext> ctx) {
    if (!ctx) return;

    // 确保解码器不在使用中
    if (ctx->in_use.load()) {
        LOG_WARN_FMT("[DecoderPool] Decoder is in use, waiting for release");
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // 显式清理解码器资源
    ctx.reset();
}

std::shared_ptr<DecoderContext> DecoderPool::createDecoder(int codec_id) {
    auto ctx = std::make_shared<DecoderContext>();
    ctx->codec_id = codec_id;

    // 尝试使用硬件解码器（CUDA）
    bool hw_accel_success = false;
    do {
        // 根据编码格式选择硬件解码器
        const AVCodec* hw_codec = nullptr;
        if (codec_id == AV_CODEC_ID_H264) {
            hw_codec = avcodec_find_decoder_by_name("h264_cuvid");
        } else if (codec_id == AV_CODEC_ID_HEVC) {
            hw_codec = avcodec_find_decoder_by_name("hevc_cuvid");
        }

        if (!hw_codec) {
            LOG_DEBUG_FMT("[DecoderPool] Hardware decoder not found for codec_id={}, trying software decoder", codec_id);
            break;
        }

        // 创建硬件设备上下文
        if (av_hwdevice_ctx_create(&ctx->hw_device_ctx, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0) < 0) {
            LOG_DEBUG_FMT("[DecoderPool] Failed to create CUDA device context");
            break;
        }

        // 分配硬件解码器上下文
        ctx->codec_ctx = avcodec_alloc_context3(hw_codec);
        if (!ctx->codec_ctx) {
            LOG_DEBUG_FMT("[DecoderPool] Failed to allocate hardware codec context");
            break;
        }

        // 设置硬件设备上下文
        ctx->codec_ctx->hw_device_ctx = av_buffer_ref(ctx->hw_device_ctx);
        ctx->hw_accel_enabled = true;

        // 打开硬件解码器
        if (avcodec_open2(ctx->codec_ctx, hw_codec, nullptr) < 0) {
            LOG_DEBUG_FMT("[DecoderPool] Failed to open hardware codec");
            break;
        }

        // 分配硬件帧
        ctx->hw_frame = av_frame_alloc();
        if (!ctx->hw_frame) {
            LOG_DEBUG_FMT("[DecoderPool] Failed to allocate hardware frame");
            break;
        }

        hw_accel_success = true;
        LOG_INFO_FMT("[DecoderPool] Hardware decoder created successfully: {}", hw_codec->name);
    } while (false);

    // 如果硬件解码失败，回退到软件解码
    if (!hw_accel_success) {
        LOG_INFO_FMT("[DecoderPool] Falling back to software decoder");

        // 清理硬件相关资源
        if (ctx->hw_device_ctx) {
            av_buffer_unref(&ctx->hw_device_ctx);
        }
        if (ctx->hw_frame) {
            av_frame_free(&ctx->hw_frame);
        }

        // 查找软件解码器
        const AVCodec* codec = avcodec_find_decoder((AVCodecID)codec_id);
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

        // 设置线程数（提高解码性能）
        ctx->codec_ctx->thread_count = 4;
        ctx->codec_ctx->thread_type = FF_THREAD_FRAME | FF_THREAD_SLICE;

        // 打开解码器
        if (avcodec_open2(ctx->codec_ctx, codec, nullptr) < 0) {
            LOG_ERROR_FMT("[DecoderPool] Failed to open codec");
            return nullptr;
        }
    }

    // 分配软件帧（用于格式转换或fallback）
    ctx->frame = av_frame_alloc();
    ctx->bgr_frame = av_frame_alloc();
    if (!ctx->frame || !ctx->bgr_frame) {
        LOG_ERROR_FMT("[DecoderPool] Failed to allocate frames");
        return nullptr;
    }

    ctx->initialized = true;
    LOG_INFO_FMT("[DecoderPool] Decoder created: {} (hw_accel={})",
                 ctx->hw_accel_enabled ? "h264_cuvid" : "software",
                 ctx->hw_accel_enabled);

    return ctx;
}

} // namespace nodes
} // namespace ai_stream