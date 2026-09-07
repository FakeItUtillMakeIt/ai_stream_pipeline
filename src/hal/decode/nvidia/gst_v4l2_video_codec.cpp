// src/hal/decode/nvidia/gst_v4l2_video_codec.cpp
// Jetson GStreamer nvv4l2 硬件解码后端实现。
#include "gst_v4l2_video_codec.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <libavutil/pixfmt.h>

#include <mutex>

namespace ai_stream {
namespace hal {

namespace {
std::once_flag g_gst_init_flag;
void ensureGstInit() {
    std::call_once(g_gst_init_flag, []() { gst_init(nullptr, nullptr); });
}
} // namespace

GstV4l2VideoCodec::GstV4l2VideoCodec() {
    LOG_DEBUG("[GstV4l2VideoCodec] Constructor");
}

GstV4l2VideoCodec::~GstV4l2VideoCodec() {
    release();
    LOG_DEBUG("[GstV4l2VideoCodec] Destroyed");
}

bool GstV4l2VideoCodec::isAvailable() const {
    ensureGstInit();
    return gst_element_factory_find("nvv4l2decoder") != nullptr &&
           gst_element_factory_find("nvvidconv") != nullptr;
}

bool GstV4l2VideoCodec::buildPipeline(const std::string& codec_name) {
    const char* parser_name = (codec_name == "h264" || codec_name == "H264")
                                  ? "h264parse"
                                  : "h265parse";

    GstElement* parser = gst_element_factory_make(parser_name, "parser");
    GstElement* dec = gst_element_factory_make("nvv4l2decoder", "dec");
    GstElement* conv = gst_element_factory_make("nvvidconv", "conv");
    GstElement* vconv = gst_element_factory_make("videoconvert", "vconv");
    appsrc_ = gst_element_factory_make("appsrc", "src");
    appsink_ = gst_element_factory_make("appsink", "sink");
    if (!parser || !dec || !conv || !vconv || !appsrc_ || !appsink_) {
        LOG_ERROR("[GstV4l2VideoCodec] Failed to create pipeline elements");
        return false;
    }
    app_src_ = GST_APP_SRC(appsrc_);
    app_sink_ = GST_APP_SINK(appsink_);

    pipeline_ = gst_pipeline_new("nvv4l2dec-pipeline");

    // appsrc 输入 caps：字节流编码数据
    std::string media = (codec_name == "h264" || codec_name == "H264") ? "video/x-h264" : "video/x-h265";
    GstCaps* src_caps = gst_caps_from_string((media + ", stream-format=(string)byte-stream").c_str());
    g_object_set(G_OBJECT(appsrc_), "caps", src_caps, "format", GST_FORMAT_TIME,
                 "is-live", FALSE, "max-bytes", (guint64)-1, nullptr);
    gst_caps_unref(src_caps);

    // nvvidconv：NVMM → 系统内存 NV12
    GstCaps* conv_caps = gst_caps_from_string(
        "video/x-raw(memory:NVMM), format=(string)NV12, width=(int)[16,8192], height=(int)[16,8192]");
    g_object_set(G_OBJECT(conv), "src-caps", conv_caps, nullptr);
    gst_caps_unref(conv_caps);

    // appsink 输出 caps：BGR24（CPU 内存）
    GstCaps* sink_caps = gst_caps_from_string(
        "video/x-raw, format=(string)BGR, width=(int)[16,8192], height=(int)[16,8192]");
    g_object_set(G_OBJECT(appsink_), "caps", sink_caps, "sync", FALSE,
                 "max-buffers", 4, "drop", FALSE, nullptr);
    gst_caps_unref(sink_caps);

    gst_bin_add_many(GST_BIN(pipeline_), appsrc_, parser, dec, conv, vconv, appsink_, nullptr);
    if (!gst_element_link_many(appsrc_, parser, dec, conv, vconv, appsink_, nullptr)) {
        LOG_ERROR("[GstV4l2VideoCodec] Failed to link pipeline");
        return false;
    }

    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("[GstV4l2VideoCodec] Failed to start pipeline");
        return false;
    }
    return true;
}

bool GstV4l2VideoCodec::init(const std::string& codec_name,
                             const uint8_t* extradata,
                             int extradata_size) {
    ensureGstInit();
    if (extradata && extradata_size > 0) {
        extradata_.assign(extradata, extradata + extradata_size);
    }
    if (!buildPipeline(codec_name)) {
        return false;
    }
    initialized_ = true;
    LOG_INFO_FMT("[GstV4l2VideoCodec] Decoder initialized: {}", codec_name);
    return true;
}

bool GstV4l2VideoCodec::pushPacket(const uint8_t* data, int size) {
    std::vector<uint8_t> tmp;
    if (!fed_extradata_ && !extradata_.empty()) {
        tmp.reserve(static_cast<size_t>(size) + extradata_.size());
        tmp.insert(tmp.end(), extradata_.begin(), extradata_.end());
        tmp.insert(tmp.end(), data, data + size);
        data = tmp.data();
        size = static_cast<int>(tmp.size());
        fed_extradata_ = true;
    }

    GstBuffer* buf = gst_buffer_new_allocate(nullptr, size, nullptr);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_WRITE);
    std::memcpy(map.data, data, size);
    gst_buffer_unmap(buf, &map);

    constexpr int kFps = 30;
    const guint64 dur = GST_SECOND / kFps;
    GST_BUFFER_PTS(buf) = pts_counter_ * dur;
    GST_BUFFER_DURATION(buf) = dur;
    pts_counter_++;

    GstFlowReturn ret = gst_app_src_push_buffer(app_src_, buf);
    if (ret != GST_FLOW_OK) {
        LOG_WARN_FMT("[GstV4l2VideoCodec] appsrc push failed: {}", static_cast<int>(ret));
        return false;
    }
    return true;
}

bool GstV4l2VideoCodec::pullFrame(DecodedFrame& frame) {
    GstSample* sample = gst_app_sink_try_pull_sample(app_sink_, 20 * GST_MSECOND);
    if (!sample) {
        return false;
    }

    GstBuffer* buf = gst_sample_get_buffer(sample);
    GstCaps* caps = gst_sample_get_caps(sample);

    GstStructure* s = caps ? gst_caps_get_structure(caps, 0) : nullptr;
    int w = 0, h = 0;
    if (!s || !gst_structure_get_int(s, "width", &w) ||
        !gst_structure_get_int(s, "height", &h) || w <= 0 || h <= 0) {
        LOG_WARN("[GstV4l2VideoCodec] Cannot parse video caps");
        gst_sample_unref(sample);
        return false;
    }
    width_ = w;
    height_ = h;
    const gsize size = gst_buffer_get_size(buf);
    if (size < static_cast<gsize>(w * h * 3)) {
        LOG_WARN_FMT("[GstV4l2VideoCodec] Unexpected buffer size {} for {}x{} BGR", size, w, h);
        gst_sample_unref(sample);
        return false;
    }

    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
        gst_sample_unref(sample);
        return false;
    }

    frame_data_.resize(size);
    std::memcpy(frame_data_.data(), map.data, size);
    frame_pitch_ = static_cast<int>(size / static_cast<gsize>(h));
    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);

    frame.data = frame_data_.data();
    frame.pitch = frame_pitch_;
    frame.width = width_;
    frame.height = height_;
    frame.format = AV_PIX_FMT_BGR24;
    frame.data_uv = nullptr;
    frame.pitch_uv = 0;
    frame.owns_data = false;
    frame.is_gpu = false;
    return true;
}

bool GstV4l2VideoCodec::decode(const uint8_t* packet_data, int packet_size,
                               DecodedFrame& frame) {
    if (!initialized_) {
        return false;
    }

    // 处理 bus 上的错误/警告
    GstBus* bus = gst_element_get_bus(pipeline_);
    if (bus) {
        GstMessage* msg = gst_bus_pop_filtered(
            bus, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_WARNING));
        if (msg) {
            GError* err = nullptr;
            gchar* dbg = nullptr;
            if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
                gst_message_parse_error(msg, &err, &dbg);
                LOG_ERROR_FMT("[GstV4l2VideoCodec] GStreamer error: {} ({})", err->message, dbg);
            } else {
                gst_message_parse_warning(msg, &err, &dbg);
                LOG_WARN_FMT("[GstV4l2VideoCodec] GStreamer warning: {}", err->message);
            }
            g_clear_error(&err);
            g_free(dbg);
            gst_message_unref(msg);
        }
        gst_object_unref(bus);
    }

    if (!pushPacket(packet_data, packet_size)) {
        return false;
    }
    return pullFrame(frame);
}

void GstV4l2VideoCodec::release() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        appsrc_ = nullptr;
        appsink_ = nullptr;
        app_src_ = nullptr;
        app_sink_ = nullptr;
    }
    initialized_ = false;
    fed_extradata_ = false;
    pts_counter_ = 0;
    width_ = 0;
    height_ = 0;
    frame_data_.clear();
    frame_pitch_ = 0;
    extradata_.clear();
}

// 注册到工厂（Jetson 平台）
REGISTER_VIDEO_CODEC(VideoCodecBackend::NVV4L2, GstV4l2VideoCodec)

} // namespace hal
} // namespace ai_stream