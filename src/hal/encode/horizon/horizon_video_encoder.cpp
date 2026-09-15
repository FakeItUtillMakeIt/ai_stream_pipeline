// src/hal/encode/horizon/horizon_video_encoder.cpp
// Horizon VPU 硬件 H.264 编码器实现（libspcdev sp_* 编码 API）
#include "horizon_video_encoder.h"
#include "ai_stream/hal/h264_extradata.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <cstring>
#include <mutex>

#define SP_ENCODER_H264 1

namespace {

// ---- sp 编码库函数表（全局共享 dlopen 句柄） ----
struct SpEncFuncs {
    void* dl = nullptr;
    std::mutex mtx;

    void* (*init_module)() = nullptr;
    void (*release_module)(void*) = nullptr;
    int32_t (*start_encode)(void*, int32_t, int32_t, int32_t, int32_t, int32_t) = nullptr;
    int32_t (*stop_encode)(void*) = nullptr;
    int32_t (*set_frame)(void*, char*, int32_t) = nullptr;
    int32_t (*get_stream)(void*, char*) = nullptr;

    bool ensure() {
        std::lock_guard<std::mutex> lk(mtx);
        if (dl) return true;
        void* d = dlopen("libspcdev.so", RTLD_NOW | RTLD_GLOBAL);
        if (!d) {
            LOG_INFO_FMT("[HorizonVideoEncoder] dlopen libspcdev.so failed: {}", dlerror());
            return false;
        }
        init_module = reinterpret_cast<decltype(init_module)>(dlsym(d, "sp_init_encoder_module"));
        release_module = reinterpret_cast<decltype(release_module)>(dlsym(d, "sp_release_encoder_module"));
        start_encode = reinterpret_cast<decltype(start_encode)>(dlsym(d, "sp_start_encode"));
        stop_encode = reinterpret_cast<decltype(stop_encode)>(dlsym(d, "sp_stop_encode"));
        set_frame = reinterpret_cast<decltype(set_frame)>(dlsym(d, "sp_encoder_set_frame"));
        get_stream = reinterpret_cast<decltype(get_stream)>(dlsym(d, "sp_encoder_get_stream"));
        if (!init_module || !release_module || !start_encode || !stop_encode ||
            !set_frame || !get_stream) {
            dlclose(d);
            init_module = nullptr;
            return false;
        }
        dl = d;
        return true;
    }
};

SpEncFuncs& funcs() {
    static SpEncFuncs f;
    return f;
}

// 扫描 AnnexB 是否含 IDR（NAL type 5）→ 关键帧
bool hasIdr(const uint8_t* d, size_t n) {
    for (size_t i = 0; i + 3 < n; ++i) {
        size_t sc = 0;
        if (i + 4 < n && d[i] == 0 && d[i+1] == 0 && d[i+2] == 0 && d[i+3] == 1) sc = 4;
        else if (d[i] == 0 && d[i+1] == 0 && d[i+2] == 1) sc = 3;
        if (!sc) continue;
        if ((d[i+sc] & 0x1F) == 5) return true;
        i += sc - 1;
    }
    return false;
}

// 从 AnnexB 码流中提取 SPS(7)/PPS(8) 作为 extradata（保留起始码）
void extractSpsPps(const uint8_t* d, size_t n, std::vector<uint8_t>& out) {
    out.clear();
    size_t i = 0;
    while (i + 3 < n) {
        size_t sc = 0;
        if (i + 4 < n && d[i] == 0 && d[i+1] == 0 && d[i+2] == 0 && d[i+3] == 1) sc = 4;
        else if (d[i] == 0 && d[i+1] == 0 && d[i+2] == 1) sc = 3;
        if (!sc) { ++i; continue; }
        size_t nal = i + sc;
        size_t j = nal + 1;
        while (j + 3 < n) {
            if ((j + 4 < n && d[j]==0 && d[j+1]==0 && d[j+2]==0 && d[j+3]==1) ||
                (d[j]==0 && d[j+1]==0 && d[j+2]==1)) break;
            ++j;
        }
        int type = d[nal] & 0x1F;
        if (type == 7 || type == 8) out.insert(out.end(), d + i, d + j);
        i = j;
    }
}

} // namespace

namespace ai_stream {
namespace hal {

HorizonVideoEncoder::HorizonVideoEncoder() = default;

HorizonVideoEncoder::~HorizonVideoEncoder() { close(); }

bool HorizonVideoEncoder::isAvailable() const { return funcs().ensure(); }

bool HorizonVideoEncoder::open(const VideoEncoderConfig& config) {
    std::lock_guard<std::mutex> lk(mutex_);
    closeUnlocked();

    if (!funcs().ensure()) return false;

    width_ = config.width;
    height_ = config.height;
    gop_ = config.gop > 0 ? config.gop : 25;
    if (width_ <= 0 || height_ <= 0 || (width_ % 2) || (height_ % 2)) {
        LOG_ERROR_FMT("[HorizonVideoEncoder] invalid size {}x{}", width_, height_);
        return false;
    }

    obj_ = funcs().init_module();
    if (!obj_) {
        LOG_ERROR("[HorizonVideoEncoder] sp_init_encoder_module failed");
        return false;
    }

    int ret = funcs().start_encode(obj_, 0, SP_ENCODER_H264, width_, height_,
                                   config.bitrate_kbps);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonVideoEncoder] sp_start_encode failed: {}", ret);
        funcs().release_module(obj_);
        obj_ = nullptr;
        return false;
    }

    nv12_.assign(static_cast<size_t>(width_) * height_ * 3 / 2, 0);
    stream_.assign(static_cast<size_t>(width_) * height_ * 2 + (1 << 20), 0);
    extradata_.clear();
    got_extradata_ = false;
    opened_ = true;

    LOG_INFO_FMT("[HorizonVideoEncoder] opened {}x{} {}kbps (VPU H.264)", width_, height_,
                 config.bitrate_kbps);
    return true;
}

bool HorizonVideoEncoder::encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                                 std::vector<EncodedPacket>& packets) {
    std::lock_guard<std::mutex> lk(mutex_);
    if (!opened_ || !obj_ || !yuv420p) return false;
    const size_t expect = static_cast<size_t>(width_) * height_ * 3 / 2;
    if (size < expect) return false;

    // I420 -> NV12（Y 原样；U/V 平面交织成 UV）
    const int w = width_, h = height_;
    uint8_t* dst = nv12_.data();
    memcpy(dst, yuv420p, static_cast<size_t>(w) * h);
    const uint8_t* u = yuv420p + static_cast<size_t>(w) * h;
    const uint8_t* v = u + static_cast<size_t>(w / 2) * (h / 2);
    uint8_t* uv = dst + static_cast<size_t>(w) * h;
    const int uv_n = (w / 2) * (h / 2);
    for (int i = 0; i < uv_n; ++i) {
        uv[2 * i] = u[i];
        uv[2 * i + 1] = v[i];
    }

    int32_t ret = funcs().set_frame(obj_, reinterpret_cast<char*>(nv12_.data()),
                                    static_cast<int32_t>(expect));
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonVideoEncoder] sp_encoder_set_frame failed: {}", ret);
        return false;
    }

    int32_t n = funcs().get_stream(obj_, reinterpret_cast<char*>(stream_.data()));
    if (n <= 0) return true;  // 编码流水线延迟，暂无输出（非错误）

    const uint8_t* p = stream_.data();
    if (!got_extradata_) {
        extractSpsPps(p, static_cast<size_t>(n), extradata_);
        if (!extradata_.empty()) got_extradata_ = true;
    }
    EncodedPacket pkt;
    pkt.data = p;
    pkt.size = static_cast<size_t>(n);
    pkt.pts = pts;
    pkt.dts = pts;
    pkt.keyframe = hasIdr(p, static_cast<size_t>(n));
    packets.push_back(pkt);
    return true;
}

void HorizonVideoEncoder::closeUnlocked() {
    if (obj_) {
        if (funcs().stop_encode) funcs().stop_encode(obj_);
        if (funcs().release_module) funcs().release_module(obj_);
        obj_ = nullptr;
    }
    opened_ = false;
    nv12_.clear();
    stream_.clear();
}

void HorizonVideoEncoder::close() {
    std::lock_guard<std::mutex> lk(mutex_);
    closeUnlocked();
}

REGISTER_VIDEO_ENCODER("horizon_h264", HorizonVideoEncoder)

} // namespace hal
} // namespace ai_stream
