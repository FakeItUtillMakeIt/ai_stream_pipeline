// src/hal/decode/nvidia/gst_v4l2_video_codec.h
// Jetson GStreamer nvv4l2 硬件解码后端。
//
// 本机 NVDEC 通过 GStreamer nvv4l2decoder 暴露（底层为 NVIDIA NvMM/GPU 路径，
// 裸 V4L2 ioctl 不可用）。管线：
//   appsrc → h265parse/h264parse → nvv4l2decoder → nvvidconv → videoconvert → appsink
// 输出 BGR24（CPU 内存），decode 节点直接拷贝为 cv::Mat。
#pragma once

#include "ai_stream/hal/i_video_codec.h"

#include <cstdint>
#include <string>
#include <vector>

typedef struct _GstElement GstElement;
typedef struct _GstAppSrc GstAppSrc;
typedef struct _GstAppSink GstAppSink;

namespace ai_stream {
namespace hal {

class GstV4l2VideoCodec : public IVideoCodec {
public:
    GstV4l2VideoCodec();
    ~GstV4l2VideoCodec() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;

    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;

    void release() override;

    std::string getName() const override { return "GStreamer nvv4l2 (Jetson HW)"; }
    bool isAvailable() const override;

private:
    bool buildPipeline(const std::string& codec_name);
    bool pushPacket(const uint8_t* data, int size);
    bool pullFrame(DecodedFrame& frame);

    GstElement* pipeline_ = nullptr;
    GstElement* appsrc_ = nullptr;
    GstElement* appsink_ = nullptr;
    GstAppSrc* app_src_ = nullptr;
    GstAppSink* app_sink_ = nullptr;

    int width_ = 0;
    int height_ = 0;
    int64_t pts_counter_ = 0;

    std::vector<uint8_t> frame_data_;   // 输出帧缓冲（跨调用复用）
    int frame_pitch_ = 0;

    std::vector<uint8_t> extradata_;
    bool fed_extradata_ = false;

    bool initialized_ = false;
};

} // namespace hal
} // namespace ai_stream