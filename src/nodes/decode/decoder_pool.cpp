// src/nodes/decode/decoder_pool.cpp
#include "decoder_pool.h"
#include "3rd_party/log_mgr/log_mgr.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace ai_stream {
namespace nodes {

DecoderContext::~DecoderContext() {
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

std::shared_ptr<DecoderContext> DecoderPool::getDecoder(uint32_t stream_id, int codec_id) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = decoders_.find(stream_id);
    if (it != decoders_.end()) {
        return it->second;
    }
    
    // 创建新的解码器
    auto ctx = createDecoder(codec_id);
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

std::shared_ptr<DecoderContext> DecoderPool::createDecoder(int codec_id) {
    auto ctx = std::make_shared<DecoderContext>();
    ctx->codec_id = codec_id;
    
    // 查找解码器
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
    
    // 分配帧
    ctx->frame = av_frame_alloc();
    ctx->bgr_frame = av_frame_alloc();
    if (!ctx->frame || !ctx->bgr_frame) {
        LOG_ERROR_FMT("[DecoderPool] Failed to allocate frames");
        return nullptr;
    }
    
    ctx->initialized = true;
    LOG_INFO_FMT("[DecoderPool] Decoder created: {}", codec->name);
    
    return ctx;
}

} // namespace nodes
} // namespace ai_stream