// src/hal/decode/vaapi/vaapi_video_decoder.h
// Intel/AMD VAAPI 硬件视频解码——封装 FFmpeg VAAPI hwaccel 到 HAL 接口
//
// 与 NVDEC 后端同构：解码帧留在 GPU（AV_PIX_FMT_VAAPI），通过
// av_hwframe_transfer_data 拷回主机侧 NV12 后交给节点。节点已有
// NV12 → BGR / NV12 直通两条路径，因此这里统一输出 NV12。
//
// 依赖：FFmpeg 编译时启用 VAAPI（Ubuntu 官方 libavcodec 默认启用）+ 内核
// i915/xe 驱动 + intel-media-va-driver（iHD）。不需要链接 libva。
#pragma once

#include "ai_stream/hal/i_video_decoder.h"
#include <string>

// 前向声明 FFmpeg 类型
struct AVCodecContext;
struct AVFrame;
struct AVPacket;
struct AVBufferRef;
struct SwsContext;

namespace ai_stream {
namespace hal {

class VaapiVideoDecoder : public IVideoDecoder {
public:
    VaapiVideoDecoder() = default;
    ~VaapiVideoDecoder() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;

    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;

    void release() override;

    std::string getName() const override { return "VAAPI (Intel/AMD iGPU)"; }
    bool isAvailable() const override;

private:
    bool initDecoder();
    // 把 FFmpeg 输出的任意软件像素格式转成 NV12（节点侧统一消费格式）
    bool ensureNv12Converter(int width, int height, int src_format);
    void cleanup();

    std::string codec_name_;
    const uint8_t* extradata_ = nullptr;
    int extradata_size_ = 0;

    AVCodecContext* codec_ctx_ = nullptr;
    AVFrame* frame_ = nullptr;      // VAAPI 硬件帧
    AVFrame* sw_frame_ = nullptr;   // 下载到主机的帧（通常是 NV12）
    AVFrame* nv12_frame_ = nullptr; // 非 NV12 时的转换目标
    AVPacket* packet_ = nullptr;
    AVBufferRef* hw_device_ctx_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;

    int sws_width_ = 0;
    int sws_height_ = 0;
    int sws_src_format_ = -1;
    bool initialized_ = false;
};

} // namespace hal
} // namespace ai_stream
