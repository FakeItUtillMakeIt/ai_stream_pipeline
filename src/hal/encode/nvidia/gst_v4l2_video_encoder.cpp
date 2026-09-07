// src/hal/encode/nvidia/gst_v4l2_video_encoder.cpp
// Jetson GStreamer nvv4l2 硬件 H.264 编码后端实现。
#include "gst_v4l2_video_encoder.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "ai_stream/hal/i_video_encoder.h"

#include <gst/gst.h>
#include <gst/app/gstappsink.h>
#include <gst/app/gstappsrc.h>

#include <mutex>
#include <utility>

namespace ai_stream {
namespace hal {

namespace {
std::once_flag g_gst_init_flag;
void ensureGstInit() {
    std::call_once(g_gst_init_flag, []() { gst_init(nullptr, nullptr); });
}
} // namespace

namespace {
bool extractCodecDataFromCaps(GstCaps* caps, std::vector<uint8_t>& extradata) {
    if (!caps || gst_caps_is_empty(caps)) return false;
    const GstStructure* s = gst_caps_get_structure(caps, 0);
    if (!s) return false;
    const GValue* value = gst_structure_get_value(s, "codec_data");
    if (!value || !GST_VALUE_HOLDS_BUFFER(value)) return false;
    GstBuffer* codec_buffer = gst_value_get_buffer(value);
    if (!codec_buffer) return false;

    const gsize size = gst_buffer_get_size(codec_buffer);
    if (size < 7) return false;

    std::vector<uint8_t> data(size);
    if (gst_buffer_extract(codec_buffer, 0, data.data(), size) != size) return false;
    if (data[0] != 1) return false; // AVCDecoderConfigurationRecord

    extradata = std::move(data);
    return true;
}
} // namespace

GstV4l2VideoEncoder::GstV4l2VideoEncoder() {
    LOG_DEBUG("[GstV4l2VideoEncoder] Constructor");
}

GstV4l2VideoEncoder::~GstV4l2VideoEncoder() {
    close();
    LOG_DEBUG("[GstV4l2VideoEncoder] Destroyed");
}

bool GstV4l2VideoEncoder::isAvailable() const {
    ensureGstInit();
    return gst_element_factory_find("nvv4l2h264enc") != nullptr &&
           gst_element_factory_find("nvvidconv") != nullptr;
}

bool GstV4l2VideoEncoder::buildPipeline() {
    GstElement* conv = gst_element_factory_make("nvvidconv", "conv");
    GstElement* enc = gst_element_factory_make("nvv4l2h264enc", "enc");
    GstElement* parse = gst_element_factory_make("h264parse", "parse");
    appsrc_ = gst_element_factory_make("appsrc", "src");
    appsink_ = gst_element_factory_make("appsink", "sink");
    if (!conv || !enc || !parse || !appsrc_ || !appsink_) {
        LOG_ERROR("[GstV4l2VideoEncoder] Failed to create pipeline elements");
        return false;
    }
    app_src_ = GST_APP_SRC(appsrc_);
    app_sink_ = GST_APP_SINK(appsink_);

    pipeline_ = gst_pipeline_new("nvv4l2enc-pipeline");

    // appsrc 输入：packed I420（YUV420P）
    char caps_str[256];
    snprintf(caps_str, sizeof(caps_str),
             "video/x-raw, format=(string)I420, width=(int)%d, height=(int)%d, "
             "framerate=(fraction)%d/1",
             width_, height_, fps_);
    GstCaps* src_caps = gst_caps_from_string(caps_str);
    g_object_set(G_OBJECT(appsrc_), "caps", src_caps, "format", GST_FORMAT_TIME,
                 "is-live", FALSE, "max-bytes", (guint64)-1, nullptr);
    gst_caps_unref(src_caps);

    // nvvidconv → NVMM NV12（编码器要求 NVMM 内存）
    char conv_caps_str[256];
    snprintf(conv_caps_str, sizeof(conv_caps_str),
             "video/x-raw(memory:NVMM), format=(string)NV12, width=(int)%d, height=(int)%d",
             width_, height_);
    GstCaps* conv_caps = gst_caps_from_string(conv_caps_str);
    g_object_set(G_OBJECT(conv), "src-caps", conv_caps, nullptr);
    gst_caps_unref(conv_caps);

    g_object_set(G_OBJECT(enc),
                 "bitrate", static_cast<guint>(bitrate_ * 1000),
                 "insert-sps-pps", TRUE,
                 nullptr);


    // MP4 所需的 H.264 AVC 表示：length-prefixed AU。
    // h264parse 会把 SPS/PPS 组成 AVCDecoderConfigurationRecord，放在
    // negotiated caps 的 codec_data 中，业务层直接读取该字段。
    GstCaps* sink_caps = gst_caps_from_string(
        "video/x-h264, stream-format=(string)avc, alignment=(string)au");
    g_object_set(G_OBJECT(appsink_), "caps", sink_caps, "sync", FALSE,
                 "max-buffers", 8, "drop", FALSE, nullptr);
    gst_caps_unref(sink_caps);

    gst_bin_add_many(GST_BIN(pipeline_), appsrc_, conv, enc, parse, appsink_, nullptr);
    if (!gst_element_link_many(appsrc_, conv, enc, parse, appsink_, nullptr)) {
        LOG_ERROR("[GstV4l2VideoEncoder] Failed to link pipeline");
        return false;
    }

    GstStateChangeReturn state_ret =
        gst_element_set_state(pipeline_, GST_STATE_PLAYING);
    if (state_ret == GST_STATE_CHANGE_FAILURE) {
        LOG_ERROR("[GstV4l2VideoEncoder] Failed to start pipeline");
        return false;
    }

    // V4L2 M2M 编码器在推入首帧前不会完成 READY→PLAYING 切换。
    // 推一帧黑帧驱动状态机，否则 gst_element_get_state 会一直停在 READY。
    {
        const size_t bytes = static_cast<size_t>(width_) * height_ * 3 / 2;
        GstBuffer* seed = gst_buffer_new_allocate(nullptr, bytes, nullptr);
        GstMapInfo seed_map;
        gst_buffer_map(seed, &seed_map, GST_MAP_WRITE);
        std::memset(seed_map.data, 0, seed_map.size);
        gst_buffer_unmap(seed, &seed_map);
        GST_BUFFER_PTS(seed) = 0;
        gst_app_src_push_buffer(app_src_, seed);
    }

    GstState current_state = GST_STATE_NULL;
    GstState pending_state = GST_STATE_VOID_PENDING;
    state_ret = gst_element_get_state(pipeline_, &current_state, &pending_state,
                                      2 * GST_SECOND);
    if (state_ret == GST_STATE_CHANGE_FAILURE || current_state != GST_STATE_PLAYING) {
        LOG_ERROR_FMT("[GstV4l2VideoEncoder] Pipeline did not reach PLAYING, state={}",
                      gst_element_state_get_name(current_state));
        return false;
    }
    return true;
}

bool GstV4l2VideoEncoder::open(const VideoEncoderConfig& config) {
    ensureGstInit();
    width_ = config.width;
    height_ = config.height;
    fps_ = config.fps > 0 ? config.fps : 25;
    bitrate_ = config.bitrate_kbps;
    if (width_ <= 0 || height_ <= 0) {
        LOG_ERROR("[GstV4l2VideoEncoder] Invalid dimensions");
        return false;
    }
    if (!buildPipeline()) {
        return false;
    }
    opened_ = true;
    next_pts_ = 0;
    last_pts_ = -1;

    // 预热：编码若干黑帧，让 h264parse 完成 codec_data 协商。
    const size_t frame_bytes = static_cast<size_t>(width_) * height_ * 3 / 2;
    std::vector<uint8_t> black(frame_bytes, 0);
    std::vector<EncodedPacket> tmp;
    for (int i = 0; i < 10 && extradata_.empty(); ++i) {
        if (!encode(black.data(), black.size(), i, tmp)) {
            LOG_WARN_FMT("[GstV4l2VideoEncoder] Warmup encode failed at frame {}", i);
        }
        tmp.clear();
        pullPackets(tmp, 200);
        tmp.clear();
    }
    if (extradata_.empty()) {
        LOG_ERROR("[GstV4l2VideoEncoder] Failed to obtain h264parse codec_data during warmup");
        close();
        return false;
    }

    // drain：预热循环可能在 extradata 拿到后立即退出，pipeline 里仍残留
    // seed 帧和早期黑帧的编码输出。推几帧 dummy + 长超时拉取，确保 encoder
    // 内部缓冲排空，避免真实画面的前几帧被黑帧"顶掉"。
    for (int i = 0; i < 5; ++i) {
        encode(black.data(), black.size(), 10 + i, tmp);
        tmp.clear();
    }
    pullPackets(tmp, 300);
    tmp.clear();
    last_pts_ = -1;

    LOG_INFO_FMT("[GstV4l2VideoEncoder] Opened: {}x{} @ {}kbps {}fps", width_, height_,
                 bitrate_, fps_);
    return true;
}

bool GstV4l2VideoEncoder::encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                                 std::vector<EncodedPacket>& packets) {
    if (!opened_) {
        return false;
    }
    const size_t expect = static_cast<size_t>(width_) * height_ * 3 / 2;
    if (size < expect || !yuv420p) {
        LOG_ERROR("[GstV4l2VideoEncoder] Invalid YUV420P input size");
        return false;
    }

    GstBuffer* buf = gst_buffer_new_allocate(nullptr, size, nullptr);
    GstMapInfo map;
    gst_buffer_map(buf, &map, GST_MAP_WRITE);
    std::memcpy(map.data, yuv420p, size);
    gst_buffer_unmap(buf, &map);

    const guint64 dur = GST_SECOND / fps_;
    GST_BUFFER_PTS(buf) = static_cast<guint64>(pts) * dur;
    GST_BUFFER_DURATION(buf) = dur;

    if (gst_app_src_push_buffer(app_src_, buf) != GST_FLOW_OK) {
        LOG_WARN("[GstV4l2VideoEncoder] appsrc push failed");
        return false;
    }

    // 拉取对应输出包（编码器有缓冲，先尝试非阻塞；无包时短等）
    return pullPackets(packets, 30);
}

bool GstV4l2VideoEncoder::pullPackets(std::vector<EncodedPacket>& packets, int64_t timeout_ms) {
    out_buffers_.clear();
    for (;;) {
        GstSample* sample =
            gst_app_sink_try_pull_sample(app_sink_, timeout_ms * GST_MSECOND);
        if (!sample) break;

        // codec_data 在 sample caps 中，不在 H.264 payload 中。
        if (extradata_.empty()) {
            GstCaps* caps = gst_sample_get_caps(sample);
            if (extractCodecDataFromCaps(caps, extradata_)) {
                LOG_INFO_FMT(
                    "[GstV4l2VideoEncoder] Captured AVCC codec_data ({} bytes)",
                    extradata_.size());
                if (caps) {
                    gchar* caps_str = gst_caps_to_string(caps);
                    if (caps_str) {
                        LOG_DEBUG_FMT(
                            "[GstV4l2VideoEncoder] negotiated caps: {}", caps_str);
                        g_free(caps_str);
                    }
                }
            }
        }

        GstBuffer* buf = gst_sample_get_buffer(sample);
        if (!buf) {
            gst_sample_unref(sample);
            continue;
        }

        GstMapInfo map;
        if (!gst_buffer_map(buf, &map, GST_MAP_READ)) {
            gst_sample_unref(sample);
            continue;
        }

        // 在 unmap 前读取元数据；不要在 unmap 后使用 map.size。
        const GstClockTime pts = GST_BUFFER_PTS(buf);
        const size_t size = map.size;
        const bool keyframe =
            !GST_BUFFER_FLAG_IS_SET(buf, GST_BUFFER_FLAG_DELTA_UNIT);

        if (size == 0) {
            gst_buffer_unmap(buf, &map);
            gst_sample_unref(sample);
            continue;
        }

        // 每个包独立持有数据副本（一次 pullPackets 调用内所有包均有效）
        out_buffers_.emplace_back(map.data, map.data + map.size);
        gst_buffer_unmap(buf, &map);

        EncodedPacket p;
        if (GST_CLOCK_TIME_IS_VALID(pts)) {
            p.pts = static_cast<int64_t>(pts / (GST_SECOND / fps_));
        } else {
            p.pts = next_pts_++;
        }

        // 该编码器输出的 DTS 不可靠（实测恒为 2×PTS，导致 muxer 报
        // "pts < dts"）。编码器为 I/P-only（无 B 帧重排），输出顺序即解码
        // 顺序，直接令 dts=pts 即可满足 muxer 的 dts≤pts 且单调要求。
        p.dts = p.pts;

        // 编码器在启动瞬态可能对同一帧输出多个缓冲（pts 重复），导致
        // muxer 报 "non monotonically increasing dts"。保证 pts 严格单调。
        if (p.pts <= last_pts_) {
            p.pts = last_pts_ + 1;
        }
        last_pts_ = p.pts;
        p.dts = p.pts;

        p.keyframe = keyframe;
        p.data = out_buffers_.back().data();
        p.size = out_buffers_.back().size();

        LOG_DEBUG_FMT(
            "[GstV4l2VideoEncoder] out AVC AU: size={} pts={} dts={} key={} extradata={}",
            size, p.pts, p.dts, p.keyframe, extradata_.size());

        packets.push_back(p);
        gst_sample_unref(sample);
    }
    return true;
}

bool GstV4l2VideoEncoder::flush(std::vector<EncodedPacket>& packets) {
    if (!opened_) {
        return true;
    }
    gst_app_src_end_of_stream(app_src_);
    return pullPackets(packets, 100);
}

void GstV4l2VideoEncoder::close() {
    if (pipeline_) {
        gst_element_set_state(pipeline_, GST_STATE_NULL);
        gst_object_unref(pipeline_);
        pipeline_ = nullptr;
        appsrc_ = nullptr;
        appsink_ = nullptr;
        app_src_ = nullptr;
        app_sink_ = nullptr;
    }
    opened_ = false;
    out_buffers_.clear();
    extradata_.clear();
}

// 注册到编码工厂（Jetson 平台）
REGISTER_VIDEO_ENCODER("nvv4l2_h264", GstV4l2VideoEncoder)

} // namespace hal
} // namespace ai_stream