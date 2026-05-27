// src/nodes/decode/decoder_pool.h
#pragma once

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
    AVCodecContext* codec_ctx = nullptr;
    AVFrame* frame = nullptr;
    AVFrame* bgr_frame = nullptr;      // 用于格式转换的帧
    SwsContext* sws_ctx = nullptr;     // 用于像素格式转换
    uint8_t* bgr_buffer = nullptr;     // BGR 格式缓冲区
    int buffer_size = 0;

    //硬件解码字段
    bool use_hw = false;
    AVBufferRef* hw_device_ctx = nullptr;  // 硬件设备上下文
    AVFrame* hw_frame = nullptr;        // 用于硬件解码的帧

    void* d_bgr_buffer = nullptr;
    size_t d_bgr_buffer_size = 0;

    int width = 0;
    int height = 0;
    int codec_id = 0;
    bool initialized = false;
    std::atomic<bool> in_use{false};
    
    ~DecoderContext();
};

/**
 * @brief 解码器池：为每个 stream_id 维护独立的解码上下文，支持多流并行
 */
class DecoderPool {
public:
    DecoderPool();
    ~DecoderPool();

    /**
     * @brief 获取或创建指定 stream_id 的解码器
     */
    std::shared_ptr<DecoderContext> getDecoder(uint32_t stream_id, int codec_id = 0,bool use_hw = false);

    /**
     * @brief 释放指定 stream_id 的解码器
     */
    void releaseDecoder(uint32_t stream_id);

    /**
     * @brief 获取当前活跃解码器数量
     */
    size_t getActiveCount() const;

    /**
     * @brief 清理所有解码器
     */
    void clear();

private:
    std::shared_ptr<DecoderContext> createDecoder(int codec_id,bool use_hw=false);
    void destroyDecoder(std::shared_ptr<DecoderContext> ctx);

    std::unordered_map<uint32_t, std::shared_ptr<DecoderContext>> decoders_;
    mutable std::mutex mutex_;
};

} // namespace nodes
} // namespace ai_stream