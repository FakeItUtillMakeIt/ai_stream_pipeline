// src/hal/decode/nvidia/gst_v4l2_video_decoder.cpp
// Jetson GStreamer nvv4l2 硬件解码后端实现。
#include "gst_v4l2_video_decoder.h"
#include "ai_stream/hal/video_decoder_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>
#include <gst/video/video.h>
#include <libavutil/pixfmt.h>

#include <mutex>

namespace ai_stream {
namespace hal {

namespace {
std::once_flag g_gst_init_flag;
void ensureGstInit() {
    std::call_once(g_gst_init_flag, []() { gst_init(nullptr, nullptr); });
}

bool hasAnnexBStartCode(const uint8_t* d, int n) {
    return n >= 4 && d[0] == 0 && d[1] == 0 && (d[2] == 1 || (d[2] == 0 && d[3] == 1));
}

// MP4/RTSP 的 H.264 包通常是 AVCC（4 字节大端长度 + NAL），
// 而 h264parse 的 byte-stream caps 需要 Annex-B 起始码，必须转换。
bool avccToAnnexB(const uint8_t* d, int n, std::vector<uint8_t>& out) {
    out.clear();
    out.reserve(static_cast<size_t>(n) + 64);
    int i = 0;
    while (i + 4 <= n) {
        const uint32_t len = (static_cast<uint32_t>(d[i]) << 24) |
                             (static_cast<uint32_t>(d[i + 1]) << 16) |
                             (static_cast<uint32_t>(d[i + 2]) << 8) |
                             static_cast<uint32_t>(d[i + 3]);
        i += 4;
        if (len == 0 || i + static_cast<int>(len) > n) {
            return false;
        }
        out.insert(out.end(), {0, 0, 0, 1});
        out.insert(out.end(), d + i, d + i + len);
        i += static_cast<int>(len);
    }
    return i == n && !out.empty();
}

// avcC (AVCDecoderConfigurationRecord) → Annex-B 的 SPS/PPS
std::vector<uint8_t> avcCToAnnexB(const std::vector<uint8_t>& ex) {
    std::vector<uint8_t> out;
    if (ex.size() < 7 || ex[0] != 1) return out;
    size_t p = 5;
    const int num_sps = ex[p++] & 0x1f;
    for (int k = 0; k < num_sps && p + 2 <= ex.size(); ++k) {
        const int len = (ex[p] << 8) | ex[p + 1];
        p += 2;
        if (p + static_cast<size_t>(len) > ex.size()) return {};
        out.insert(out.end(), {0, 0, 0, 1});
        out.insert(out.end(), ex.begin() + p, ex.begin() + p + len);
        p += len;
    }
    if (p >= ex.size()) return out;
    const int num_pps = ex[p++];
    for (int k = 0; k < num_pps && p + 2 <= ex.size(); ++k) {
        const int len = (ex[p] << 8) | ex[p + 1];
        p += 2;
        if (p + static_cast<size_t>(len) > ex.size()) return {};
        out.insert(out.end(), {0, 0, 0, 1});
        out.insert(out.end(), ex.begin() + p, ex.begin() + p + len);
        p += len;
    }
    return out;
}
} // namespace

GstV4l2VideoDecoder::GstV4l2VideoDecoder() {
    LOG_DEBUG("[GstV4l2VideoDecoder] Constructor");
}

GstV4l2VideoDecoder::~GstV4l2VideoDecoder() {
    release();
    LOG_DEBUG("[GstV4l2VideoDecoder] Destroyed");
}

bool GstV4l2VideoDecoder::isAvailable() const {
    ensureGstInit();
    return gst_element_factory_find("nvv4l2decoder") != nullptr &&
           gst_element_factory_find("nvvidconv") != nullptr;
}

bool GstV4l2VideoDecoder::buildPipeline(const std::string& codec_name) {
    const char* parser_name = (codec_name == "h264" || codec_name == "H264")
                                  ? "h264parse"
                                  : "h265parse";

    GstElement* parser = gst_element_factory_make(parser_name, "parser");
    GstElement* dec = gst_element_factory_make("nvv4l2decoder", "dec");
    GstElement* conv = gst_element_factory_make("nvvidconv", "conv");
    GstElement* conv_caps_filter = gst_element_factory_make("capsfilter", "conv_caps");
    appsrc_ = gst_element_factory_make("appsrc", "src");
    appsink_ = gst_element_factory_make("appsink", "sink");
    if (!parser || !dec || !conv || !conv_caps_filter || !appsrc_ || !appsink_) {
        LOG_ERROR("[GstV4l2VideoDecoder] Failed to create pipeline elements");
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

    // nvvidconv 默认输出可能是 NVMM 显存，必须在 nvvidconv 之后用 capsfilter
    // 显式要求系统内存 NV12。
    // 直接输出 NV12（不再做 videoconvert→BGR）：后续由 resize_normalize
    // 走 NPP NV12→BGR + resize + normalize，或由解码节点按需转换。
    GstCaps* conv_caps = gst_caps_from_string(
        "video/x-raw(memory:SystemMemory), format=(string)NV12");
    g_object_set(G_OBJECT(conv_caps_filter), "caps", conv_caps, nullptr);
    gst_caps_unref(conv_caps);

    // appsink 输出 caps：系统内存 NV12
    GstCaps* sink_caps = gst_caps_from_string(
        "video/x-raw, format=(string)NV12, width=(int)[16,8192], height=(int)[16,8192]");
    g_object_set(G_OBJECT(appsink_), "caps", sink_caps, "sync", FALSE,
                 "max-buffers", 4, "drop", FALSE, nullptr);
    gst_caps_unref(sink_caps);

    gst_bin_add_many(GST_BIN(pipeline_), appsrc_, parser, dec, conv, conv_caps_filter,
                     appsink_, nullptr);
    if (!gst_element_link_many(appsrc_, parser, dec, conv, conv_caps_filter,
                               appsink_, nullptr)) {
        LOG_ERROR("[GstV4l2VideoDecoder] Failed to link pipeline");
        return false;
    }

    if (gst_element_set_state(pipeline_, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("[GstV4l2VideoDecoder] Failed to start pipeline");
        return false;
    }
    return true;
}

bool GstV4l2VideoDecoder::init(const std::string& codec_name,
                             const uint8_t* extradata,
                             int extradata_size) {
    ensureGstInit();
    is_h264_ = (codec_name == "h264" || codec_name == "H264");
    if (extradata && extradata_size > 0) {
        extradata_.assign(extradata, extradata + extradata_size);
    }
    if (!buildPipeline(codec_name)) {
        return false;
    }
    initialized_ = true;
    LOG_INFO_FMT("[GstV4l2VideoDecoder] Decoder initialized: {}", codec_name);
    return true;
}

bool GstV4l2VideoDecoder::pushPacket(const uint8_t* data, int size) {
    if (!data || size <= 0) return false;

    std::vector<uint8_t> tmp;
    if (hasAnnexBStartCode(data, size)) {
        // 已是 Annex-B：首包前补 SPS/PPS（avcC 需转成 Annex-B）
        if (!fed_extradata_ && !extradata_.empty()) {
            auto ex = is_h264_ ? avcCToAnnexB(extradata_) : std::vector<uint8_t>{};
            if (!ex.empty()) {
                tmp.assign(ex.begin(), ex.end());
            }
            fed_extradata_ = true;
        }
        if (!tmp.empty()) {
            tmp.insert(tmp.end(), data, data + size);
            data = tmp.data();
            size = static_cast<int>(tmp.size());
        }
    } else {
        // AVCC（4 字节长度前缀）→ Annex-B
        std::vector<uint8_t> converted;
        if (!avccToAnnexB(data, size, converted)) {
            // 解析失败则原样发送，避免整条流直接停止
            converted.assign(data, data + size);
        }
        if (!fed_extradata_ && !extradata_.empty()) {
            auto ex = is_h264_ ? avcCToAnnexB(extradata_) : std::vector<uint8_t>{};
            if (!ex.empty()) {
                converted.insert(converted.begin(), ex.begin(), ex.end());
            }
            fed_extradata_ = true;
        }
        tmp.swap(converted);
        data = tmp.data();
        size = static_cast<int>(tmp.size());
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
        LOG_WARN_FMT("[GstV4l2VideoDecoder] appsrc push failed: {}", static_cast<int>(ret));
        return false;
    }
    return true;
}

bool GstV4l2VideoDecoder::pullFrame(DecodedFrame& frame) {
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
        LOG_WARN("[GstV4l2VideoDecoder] Cannot parse video caps");
        gst_sample_unref(sample);
        return false;
    }
    width_ = w;
    height_ = h;

    // 从 caps 取 NV12 的 plane offset/stride（nvvidconv 输出可能带行对齐）
    int y_offset = 0, y_pitch = w;
    int uv_offset = w * h, uv_pitch = w;
    GstVideoInfo info;
    if (caps && gst_video_info_from_caps(&info, caps) &&
        GST_VIDEO_INFO_N_PLANES(&info) >= 2) {
        y_offset = static_cast<int>(GST_VIDEO_INFO_PLANE_OFFSET(&info, 0));
        y_pitch = static_cast<int>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 0));
        uv_offset = static_cast<int>(GST_VIDEO_INFO_PLANE_OFFSET(&info, 1));
        uv_pitch = static_cast<int>(GST_VIDEO_INFO_PLANE_STRIDE(&info, 1));
    }

    const gsize size = gst_buffer_get_size(buf);
    const gsize need = static_cast<gsize>(uv_offset) +
                       static_cast<gsize>(uv_pitch) * static_cast<gsize>(h / 2);
    if (size < need) {
        LOG_WARN_FMT("[GstV4l2VideoDecoder] Unexpected NV12 buffer size {} for {}x{} (need {})",
                     size, w, h, need);
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
    gst_buffer_unmap(buf, &map);
    gst_sample_unref(sample);

    frame.data = frame_data_.data() + y_offset;
    frame.pitch = y_pitch;
    frame.width = width_;
    frame.height = height_;
    frame.format = AV_PIX_FMT_NV12;
    frame.data_uv = frame_data_.data() + uv_offset;
    frame.pitch_uv = uv_pitch;
    frame.owns_data = false;
    frame.is_gpu = false;
    return true;
}

bool GstV4l2VideoDecoder::decode(const uint8_t* packet_data, int packet_size,
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
                LOG_ERROR_FMT("[GstV4l2VideoDecoder] GStreamer error: {} ({})", err->message, dbg);
            } else {
                gst_message_parse_warning(msg, &err, &dbg);
                LOG_WARN_FMT("[GstV4l2VideoDecoder] GStreamer warning: {}", err->message);
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

void GstV4l2VideoDecoder::release() {
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
REGISTER_VIDEO_DECODER(VideoDecoderBackend::NVV4L2, GstV4l2VideoDecoder)

} // namespace hal
} // namespace ai_stream