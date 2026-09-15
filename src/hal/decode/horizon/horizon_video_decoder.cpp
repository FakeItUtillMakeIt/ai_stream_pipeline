// src/hal/decode/horizon/horizon_video_decoder.cpp
// Horizon VPU 解码器实现——RDK S100P
// 优先 libspcdev 的 sp_* 解码 API → FFmpeg 软件解码
//
// sp_* API（/usr/include/sp_codec.h, libspcdev.so）是地平线官方文档推荐的解码
// 接口，内部封装 hb_mm_mc，负责码流缓冲/帧重排序，直接输出 NV12 帧：
//   sp_init_decoder_module / sp_start_decode / sp_decoder_set_image /
//   sp_decoder_get_image / sp_stop_decode / sp_release_decoder_module
// 注意：H264/H265 需先喂 3~5 帧让解码器填满内部帧缓冲，之后才能取出解码帧。
#include "horizon_video_decoder.h"
#include "ai_stream/hal/video_decoder_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <atomic>
#include <algorithm>

extern "C" {
#include <libavcodec/avcodec.h>
}

// sp_codec.h 中的编码类型常量（避免依赖头文件路径）
#define SP_ENCODER_H264  1
#define SP_ENCODER_H265  2

// DecodedFrame.format 使用 FFmpeg AVPixelFormat 数值（与其它硬件后端一致），
// 下游 ffmpeg_decode 节点据此选择 NV12→BGR 转换路径。
#ifndef HORIZON_PIX_FMT_NV12
#define HORIZON_PIX_FMT_NV12 AV_PIX_FMT_NV12
#endif

// 解码器暖机帧数：sp 解码器需先喂入若干帧才能取到输出
static constexpr int kSpWarmupFrames = 5;

namespace {

// ---- sp 解码库函数表（全局共享 dlopen 句柄，引用计数） ----
struct SpFuncs {
    void* dl = nullptr;
    std::atomic<int> refcount{0};

    void* (*init_module)() = nullptr;
    void (*release_module)(void*) = nullptr;
    int32_t (*start_decode)(void*, const char*, int32_t, int32_t, int32_t, int32_t) = nullptr;
    int32_t (*get_image)(void*, char*) = nullptr;
    int32_t (*set_image)(void*, char*, int32_t, int32_t, int32_t) = nullptr;
    int32_t (*stop_decode)(void*) = nullptr;

    void reset() {
        if (dl) { dlclose(dl); dl = nullptr; }
        init_module = nullptr; release_module = nullptr; start_decode = nullptr;
        get_image = nullptr; set_image = nullptr; stop_decode = nullptr;
    }
};

struct FfmpegFuncs {
    void* dl = nullptr;
    std::atomic<int> refcount{0};

    const AVCodec* (*find_decoder_by_name)(const char*) = nullptr;
    AVCodecContext* (*alloc_context3)(const AVCodec*) = nullptr;
    void (*free_context)(AVCodecContext**) = nullptr;
    int (*open2)(AVCodecContext*, const AVCodec*, void**) = nullptr;
    int (*send_packet)(AVCodecContext*, const AVPacket*) = nullptr;
    int (*receive_frame)(AVCodecContext*, AVFrame*) = nullptr;
    AVFrame* (*frame_alloc)() = nullptr;
    void (*frame_free)(AVFrame**) = nullptr;
    AVPacket* (*packet_alloc)() = nullptr;
    void (*packet_free)(AVPacket**) = nullptr;
    int (*new_packet)(AVPacket*, int) = nullptr;  // av_new_packet

    void reset() {
        if (dl) { dlclose(dl); dl = nullptr; }
        find_decoder_by_name = nullptr; alloc_context3 = nullptr; free_context = nullptr;
        open2 = nullptr; send_packet = nullptr; receive_frame = nullptr;
        frame_alloc = nullptr; frame_free = nullptr; packet_alloc = nullptr;
        packet_free = nullptr; new_packet = nullptr;
    }
};

SpFuncs g_sp{};
FfmpegFuncs g_ff{};

bool spAcquire() {
    if (g_sp.refcount.load() > 0) { g_sp.refcount.fetch_add(1); return true; }

    void* dl = dlopen("libspcdev.so", RTLD_NOW | RTLD_GLOBAL);
    if (!dl) {
        LOG_INFO_FMT("[HorizonVideoDecoder] dlopen libspcdev.so failed: {}", dlerror());
        return false;
    }
    g_sp.dl = dl;

    #define L(f, s) g_sp.f = reinterpret_cast<decltype(g_sp.f)>(dlsym(dl, s))
    L(init_module, "sp_init_decoder_module");
    L(release_module, "sp_release_decoder_module");
    L(start_decode, "sp_start_decode");
    L(get_image, "sp_decoder_get_image");
    L(set_image, "sp_decoder_set_image");
    L(stop_decode, "sp_stop_decode");
    #undef L

    if (!g_sp.init_module || !g_sp.release_module || !g_sp.start_decode ||
        !g_sp.get_image || !g_sp.set_image || !g_sp.stop_decode) {
        g_sp.reset();
        return false;
    }
    g_sp.refcount.store(1);
    return true;
}

void spRelease() {
    if (g_sp.refcount.fetch_sub(1) > 1) return;
    g_sp.reset();
}

bool ffAcquire() {
    if (g_ff.refcount.load() > 0) { g_ff.refcount.fetch_add(1); return true; }

    g_ff.dl = dlopen("libavcodec.so", RTLD_NOW);
    if (!g_ff.dl) return false;

    #define L(f, s) g_ff.f = reinterpret_cast<decltype(g_ff.f)>(dlsym(g_ff.dl, s))
    L(find_decoder_by_name, "avcodec_find_decoder_by_name");
    L(alloc_context3, "avcodec_alloc_context3");
    L(free_context, "avcodec_free_context");
    L(open2, "avcodec_open2");
    L(send_packet, "avcodec_send_packet");
    L(receive_frame, "avcodec_receive_frame");
    L(frame_alloc, "av_frame_alloc");
    L(frame_free, "av_frame_free");
    L(packet_alloc, "av_packet_alloc");
    L(packet_free, "av_packet_free");
    L(new_packet, "av_new_packet");
    #undef L

    if (!g_ff.find_decoder_by_name || !g_ff.open2 || !g_ff.send_packet ||
        !g_ff.receive_frame || !g_ff.frame_alloc || !g_ff.new_packet) {
        g_ff.reset();
        return false;
    }
    g_ff.refcount.store(1);
    return true;
}

void ffRelease() {
    if (g_ff.refcount.fetch_sub(1) > 1) return;
    g_ff.reset();
}

// 把一帧 NV12（可能带 stride）紧凑拷贝进 malloc 缓冲并填充 DecodedFrame
bool fillFrameNv12(const void* y_src, int y_stride, const void* uv_src,
                   int uv_stride, int w, int h, ai_stream::hal::DecodedFrame& frame) {
    if (!y_src || !uv_src || w <= 0 || h <= 0 || y_stride < w || uv_stride < w)
        return false;
    int y_sz = w * h;
    uint8_t* buf = static_cast<uint8_t*>(malloc(y_sz + y_sz / 2));
    if (!buf) return false;

    if (y_stride == w)
        memcpy(buf, y_src, y_sz);
    else
        for (int i = 0; i < h; ++i)
            memcpy(buf + i * w, static_cast<const uint8_t*>(y_src) + i * y_stride, w);

    uint8_t* uv = buf + y_sz;
    if (uv_stride == w)
        memcpy(uv, uv_src, y_sz / 2);
    else
        for (int i = 0; i < h / 2; ++i)
            memcpy(uv + i * w, static_cast<const uint8_t*>(uv_src) + i * uv_stride, w);

    frame.data = buf;
    frame.width = w;
    frame.height = h;
    frame.pitch = w;
    frame.format = HORIZON_PIX_FMT_NV12;
    frame.owns_data = true;
    frame.data_uv = uv;
    frame.pitch_uv = w;
    return true;
}

} // anon namespace

namespace {

// 扫描 Annex-B 码流是否包含 IRAP (HEVC 16..21) NAL —— 硬解必须从随机访问点开始
bool packetHasIrap(const uint8_t* data, int size) {
    for (int i = 0; i + 3 < size; ++i) {
        int sc = 0;
        if (data[i]==0 && data[i+1]==0 && data[i+2]==0 && i+4 < size && data[i+3]==1) sc = 4;
        else if (data[i]==0 && data[i+1]==0 && data[i+2]==1) sc = 3;
        if (!sc) continue;
        int nt = (data[i+sc] >> 1) & 0x3F;
        if (nt >= 16 && nt <= 21) return true;
        i += sc - 1;
    }
    return false;
}

} // anon namespace

namespace ai_stream {
namespace hal {

HorizonVideoDecoder::HorizonVideoDecoder() = default;

HorizonVideoDecoder::~HorizonVideoDecoder() { release(); }

bool HorizonVideoDecoder::isAvailable() const {
    return spAcquire() || ffAcquire();
}

void HorizonVideoDecoder::setSourceResolution(int width, int height) {
    src_width_ = width;
    src_height_ = height;
}

bool HorizonVideoDecoder::init(const std::string& codec_name,
                             const uint8_t* extradata, int extradata_size) {
    release();
    if (tryVpDecode(codec_name, extradata, extradata_size)) {
        initialized_ = true;
        return true;
    }
    if (tryFfmpegDecode(codec_name, extradata, extradata_size)) {
        initialized_ = true;
        return true;
    }
    return false;
}

bool HorizonVideoDecoder::tryVpDecode(const std::string& codec_name,
                                    const uint8_t* extradata, int extradata_size) {
    if (getenv("FORCE_SW_DECODE")) {
        LOG_INFO("[HorizonVideoDecoder] FORCE_SW_DECODE set, skipping VPU");
        return false;
    }
    if (!spAcquire()) {
        LOG_INFO("[HorizonVideoDecoder] libspcdev.so not available");
        return false;
    }

    if (codec_name == "h264" || codec_name == "H264")
        vp_codec_type_ = SP_ENCODER_H264;
    else if (codec_name == "h265" || codec_name == "HEVC" || codec_name == "hevc")
        vp_codec_type_ = SP_ENCODER_H265;
    else {
        spRelease();
        return false;
    }

    if (extradata && extradata_size > 0)
        pending_extradata_.assign(extradata, extradata + extradata_size);

    use_vp_ = true;
    name_ = "Horizon VPU HW (sp_" + codec_name + ")";
    LOG_INFO_FMT("[HorizonVideoDecoder] sp decoder selected, type={} {}", vp_codec_type_, codec_name);

    // 若分辨率已知则立即启动；否则推迟到首帧 decode_vp
    if (src_width_ > 0 && src_height_ > 0 && !startSpDecoder()) {
        spRelease();
        use_vp_ = false;
        return false;
    }
    return true;
}

bool HorizonVideoDecoder::startSpDecoder() {
    if (sp_obj_) return true;
    if (src_width_ <= 0 || src_height_ <= 0) {
        LOG_INFO("[HorizonVideoDecoder] startSpDecoder: resolution unknown");
        return false;
    }

    sp_obj_ = g_sp.init_module();
    if (!sp_obj_) {
        LOG_INFO("[HorizonVideoDecoder] sp_init_decoder_module failed");
        return false;
    }

    int ret = g_sp.start_decode(sp_obj_, "", 0, vp_codec_type_, src_width_, src_height_);
    if (ret != 0) {
        LOG_INFO_FMT("[HorizonVideoDecoder] sp_start_decode failed: {}", ret);
        g_sp.release_module(sp_obj_);
        sp_obj_ = nullptr;
        return false;
    }

    sp_out_.assign(static_cast<size_t>(src_width_) * src_height_ * 3 / 2, 0);
    vp_fed_ = 0;
    LOG_INFO_FMT("[HorizonVideoDecoder] sp decoder started {}x{}", src_width_, src_height_);
    return true;
}

bool HorizonVideoDecoder::decode(const uint8_t* packet_data, int packet_size,
                               DecodedFrame& frame) {
    if (!initialized_ || !packet_data || packet_size <= 0) return false;

    if (use_vp_) {
        if (!sp_obj_ && !startSpDecoder()) return false;

        // 硬解必须从 IRAP (IDR/CRA) 开始：跳过 GOP 中间的非随机访问帧，
        // 否则解码器无法建立参考帧，永远不会输出。
        if (!saw_irap_) {
            if (!packetHasIrap(packet_data, packet_size)) {
                return false;
            }
            saw_irap_ = true;
            LOG_INFO("[HorizonVideoDecoder] reached random access point, starting feed");
        }

        // 首个 AU：前置 extradata (VPS/SPS/PPS)，形成随机访问点
        if (!extradata_sent_ && !pending_extradata_.empty()) {
            extradata_sent_ = true;
            std::vector<uint8_t> combined;
            combined.reserve(pending_extradata_.size() + packet_size);
            combined.insert(combined.end(), pending_extradata_.begin(), pending_extradata_.end());
            combined.insert(combined.end(), packet_data, packet_data + packet_size);
            return decode_vp(combined.data(), static_cast<int>(combined.size()), frame);
        }
        extradata_sent_ = true;
        return decode_vp(packet_data, packet_size, frame);
    }

    // ---- FFmpeg 路径 ----
    AVPacket* pkt = g_ff.packet_alloc();
    if (!pkt) return false;
    if (g_ff.new_packet(pkt, packet_size) < 0) {
        g_ff.packet_free(&pkt);
        return false;
    }
    memcpy(pkt->data, packet_data, packet_size);

    int ret = g_ff.send_packet(ff_ctx_, pkt);
    g_ff.packet_free(&pkt);

    if (ret == AVERROR(EAGAIN)) {
        if (!drainFfmpeg()) return false;
        pkt = g_ff.packet_alloc();
        if (!pkt) return false;
        if (g_ff.new_packet(pkt, packet_size) < 0) {
            g_ff.packet_free(&pkt);
            return false;
        }
        memcpy(pkt->data, packet_data, packet_size);
        ret = g_ff.send_packet(ff_ctx_, pkt);
        g_ff.packet_free(&pkt);
    }
    if (ret < 0) return false;

    drainFfmpeg();
    return popFfmpegFrame(frame);
}

bool HorizonVideoDecoder::decode_vp(const uint8_t* data, int size, DecodedFrame& frame) {
    int ret = g_sp.set_image(sp_obj_, const_cast<char*>(reinterpret_cast<const char*>(data)),
                             0, size, 0);
    if (ret != 0) {
        LOG_INFO_FMT("[HorizonVideoDecoder] sp_decoder_set_image failed: {}", ret);
        return false;
    }
    vp_fed_++;

    // 暖机阶段：解码器需要若干帧填充内部缓冲，此阶段不取输出（避免阻塞）
    if (vp_fed_ <= kSpWarmupFrames)
        return false;

    memset(sp_out_.data(), 0, sp_out_.size());
    ret = g_sp.get_image(sp_obj_, reinterpret_cast<char*>(sp_out_.data()));
    if (ret != 0)
        return false;   // 暂无可用输出（-1），合法情况

    int w = src_width_, h = src_height_;
    int y_sz = w * h;
    uint8_t* buf = static_cast<uint8_t*>(malloc(y_sz + y_sz / 2));
    if (!buf) return false;
    memcpy(buf, sp_out_.data(), y_sz + y_sz / 2);   // sp 输出为紧凑 NV12

    frame.data = buf;
    frame.data_uv = buf + y_sz;
    frame.width = w;
    frame.height = h;
    frame.pitch = w;
    frame.pitch_uv = w;
    frame.format = HORIZON_PIX_FMT_NV12;
    frame.owns_data = true;
    return true;
}

bool HorizonVideoDecoder::drainFfmpeg() {
    bool got = false;
    while (true) {
        AVFrame* f = g_ff.frame_alloc();
        if (!f) return false;
        int ret = g_ff.receive_frame(ff_ctx_, f);
        if (ret == 0) {
            ff_pending_.push_back(f);
            got = true;
        } else {
            g_ff.frame_free(&f);
            break;
        }
    }
    return got;
}

bool HorizonVideoDecoder::popFfmpegFrame(DecodedFrame& frame) {
    if (ff_pending_.empty()) return false;
    AVFrame* f = ff_pending_.front();
    ff_pending_.pop_front();
    bool ok = fillFrameNv12(f->data[0], f->linesize[0], f->data[1],
                            f->linesize[1], f->width, f->height, frame);
    g_ff.frame_free(&f);
    return ok;
}

bool HorizonVideoDecoder::tryFfmpegDecode(const std::string& codec_name,
                                        const uint8_t* extradata,
                                        int extradata_size) {
    if (!ffAcquire()) return false;

    const char* name = nullptr;
    if (codec_name == "h264" || codec_name == "H264") name = "h264";
    else if (codec_name == "h265" || codec_name == "HEVC" || codec_name == "hevc") name = "hevc";
    else return false;

    const AVCodec* codec = g_ff.find_decoder_by_name(name);
    if (!codec) return false;

    ff_ctx_ = g_ff.alloc_context3(codec);
    if (!ff_ctx_) return false;

    if (extradata && extradata_size > 0) {
        size_t cap = (static_cast<size_t>(extradata_size) + AV_INPUT_BUFFER_PADDING_SIZE + 63) & ~63UL;
        uint8_t* ed = nullptr;
        if (posix_memalign(reinterpret_cast<void**>(&ed), 64, cap) != 0) {
            g_ff.free_context(&ff_ctx_);
            return false;
        }
        memset(ed, 0, cap);
        memcpy(ed, extradata, extradata_size);
        ff_ctx_->extradata = ed;
        ff_ctx_->extradata_size = extradata_size;
    }

    if (g_ff.open2(ff_ctx_, codec, nullptr) < 0) {
        g_ff.free_context(&ff_ctx_);
        return false;
    }

    use_vp_ = false;
    name_ = "Horizon FFmpeg (CPU)";
    LOG_INFO("[HorizonVideoDecoder] Using FFmpeg SW decode");
    return true;
}

void HorizonVideoDecoder::release() {
    for (AVFrame* f : ff_pending_) g_ff.frame_free(&f);
    ff_pending_.clear();

    if (sp_obj_) {
        if (g_sp.stop_decode) g_sp.stop_decode(sp_obj_);
        if (g_sp.release_module) g_sp.release_module(sp_obj_);
        sp_obj_ = nullptr;
    }
    sp_out_.clear();
    vp_fed_ = 0;

    if (ff_ctx_ && g_ff.free_context) g_ff.free_context(&ff_ctx_);
    ff_ctx_ = nullptr;

    if (g_sp.refcount.load() > 0) spRelease();
    if (g_ff.refcount.load() > 0) ffRelease();

    pending_extradata_.clear();
    extradata_sent_ = false;
    saw_irap_ = false;
    use_vp_ = false;
    initialized_ = false;
}

REGISTER_VIDEO_DECODER(VideoDecoderBackend::HORIZON, HorizonVideoDecoder)

} // namespace hal
} // namespace ai_stream
