// src/hal/decode/horizon/horizon_video_decoder.h
// Horizon VPU 解码器——RDK S100P
// 优先 libspcdev (sp_* 解码 API) → FFmpeg 软件解码
#pragma once

#include "ai_stream/hal/i_video_decoder.h"
#include <string>
#include <vector>
#include <deque>

struct AVCodecContext;
struct AVFrame;

namespace ai_stream {
namespace hal {

class HorizonVideoDecoder : public IVideoDecoder {
public:
    HorizonVideoDecoder();
    ~HorizonVideoDecoder() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;
    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;
    void release() override;
    std::string getName() const override { return name_; }
    bool isAvailable() const override;
    void setSourceResolution(int width, int height) override;

private:
    bool tryVpDecode(const std::string& codec_name,
                     const uint8_t* extradata, int extradata_size);
    bool startSpDecoder();                      // 启动 sp 解码通道（需已知分辨率）
    bool decode_vp(const uint8_t* data, int size, DecodedFrame& frame);
    bool tryFfmpegDecode(const std::string& codec_name,
                         const uint8_t* extradata, int extradata_size);
    bool drainFfmpeg();                         // 把解码器里积压的帧全部取出到 ff_pending_
    bool popFfmpegFrame(DecodedFrame& frame);   // 从队列取一帧并转为 NV12

    // ---- sp_* 硬件解码 ----
    void* sp_obj_ = nullptr;                    // sp 解码模块对象
    std::vector<uint8_t> sp_out_;               // sp 输出 NV12 缓冲 (w*h*3/2)
    int src_width_ = 0;
    int src_height_ = 0;
    int vp_codec_type_ = 2;                     // SP_ENCODER_H265=2 / H264=1
    int vp_fed_ = 0;                            // 已喂入帧数（用于暖机）
    std::vector<uint8_t> pending_extradata_;    // 首个 AU 前置参数集
    bool extradata_sent_ = false;
    bool saw_irap_ = false;                      // 是否已到达随机访问点 (IRAP)

    // ---- FFmpeg 软件回退 ----
    AVCodecContext* ff_ctx_ = nullptr;
    std::deque<AVFrame*> ff_pending_;

    bool use_vp_ = false;
    bool initialized_ = false;
    std::string name_ = "Horizon VPU HW";
};

} // namespace hal
} // namespace ai_stream
