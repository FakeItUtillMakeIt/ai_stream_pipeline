// src/nodes/decode/decoder_pool.h
// 解码器上下文——封装 HAL VideoCodec 接口
#pragma once

#include "ai_stream/hal/i_video_codec.h"
#include <unordered_map>
#include <memory>
#include <mutex>
#include <atomic>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

struct AVCodecContext;
struct AVFrame;
struct SwsContext;
struct AVPacket;

namespace ai_stream {
namespace nodes {

/**
 * @brief 单个解码器上下文封装
 */
struct DecoderContext {
    // HAL 视频编解码器
    hal::VideoCodecPtr codec;

    // 格式转换相关（用于 CPU 路径的像素格式转换）
    AVFrame* bgr_frame = nullptr;
    SwsContext* sws_ctx = nullptr;
    uint8_t* bgr_buffer = nullptr;
    int buffer_size = 0;

    uint32_t stream_id = 0;
    int width = 0;
    int height = 0;
    bool initialized = false;
    std::atomic<bool> in_use{false};

    DecoderContext();
    ~DecoderContext();
};

} // namespace nodes
} // namespace ai_stream
