// src/hal/mpp/mpp_video_encoder.cpp
// Rockchip MPP 硬件 H.264 编码器实现——见头文件说明
#include "mpp_video_encoder.h"
#include "ai_stream/hal/i_video_encoder.h"
#include "h264_extradata.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <algorithm>
#include <cstring>

#ifndef MPP_ALIGN
#define MPP_ALIGN(x, a) (((x) + (a) - 1) & ~((a) - 1))
#endif

namespace ai_stream {
namespace hal {

namespace {

// ---- dlopen 桩（与 rknn 系列后端相同模式）----
using mpp_create_fn = MPP_RET (*)(MppCtx*, MppApi**);
using mpp_init_fn = MPP_RET (*)(MppCtx, MppCtxType, MppCodingType);
using mpp_destroy_fn = MPP_RET (*)(MppCtx);
using mpp_enc_cfg_init_fn = MPP_RET (*)(MppEncCfg*);
using mpp_enc_cfg_deinit_fn = MPP_RET (*)(MppEncCfg);
using mpp_enc_cfg_set_s32_fn = MPP_RET (*)(MppEncCfg, const char*, RK_S32);
using mpp_enc_cfg_set_u32_fn = MPP_RET (*)(MppEncCfg, const char*, RK_U32);
using mpp_buffer_group_get_fn = MPP_RET (*)(MppBufferGroup*, MppBufferType,
                                            MppBufferMode, const char*, const char*);
using mpp_buffer_group_put_fn = MPP_RET (*)(MppBufferGroup);
using mpp_buffer_get_with_tag_fn = MPP_RET (*)(MppBufferGroup, MppBuffer*, size_t,
                                               const char*, const char*);
using mpp_buffer_put_fn = MPP_RET (*)(MppBuffer);
using mpp_buffer_get_ptr_with_caller_fn = void* (*)(MppBuffer, const char*);
using mpp_frame_init_fn = MPP_RET (*)(MppFrame*);
using mpp_frame_deinit_fn = MPP_RET (*)(MppFrame*);
using mpp_frame_set_width_fn = void (*)(MppFrame, RK_U32);
using mpp_frame_set_height_fn = void (*)(MppFrame, RK_U32);
using mpp_frame_set_hor_stride_fn = void (*)(MppFrame, RK_U32);
using mpp_frame_set_ver_stride_fn = void (*)(MppFrame, RK_U32);
using mpp_frame_set_fmt_fn = void (*)(MppFrame, MppFrameFormat);
using mpp_frame_set_buffer_fn = void (*)(MppFrame, MppBuffer);
using mpp_frame_set_pts_fn = void (*)(MppFrame, RK_S64);
using mpp_packet_get_data_fn = void* (*)(const MppPacket);
using mpp_packet_get_length_fn = size_t (*)(const MppPacket);
using mpp_packet_get_pts_fn = RK_S64 (*)(const MppPacket);
using mpp_packet_get_eos_fn = RK_U32 (*)(MppPacket);
using mpp_packet_deinit_fn = MPP_RET (*)(MppPacket*);

mpp_create_fn p_create = nullptr;
mpp_init_fn p_init = nullptr;
mpp_destroy_fn p_destroy = nullptr;
mpp_enc_cfg_init_fn p_cfg_init = nullptr;
mpp_enc_cfg_deinit_fn p_cfg_deinit = nullptr;
mpp_enc_cfg_set_s32_fn p_cfg_s32 = nullptr;
mpp_enc_cfg_set_u32_fn p_cfg_u32 = nullptr;
mpp_buffer_group_get_fn p_buf_group_get = nullptr;
mpp_buffer_group_put_fn p_buf_group_put = nullptr;
mpp_buffer_get_with_tag_fn p_buf_get = nullptr;
mpp_buffer_put_fn p_buf_put = nullptr;
mpp_buffer_get_ptr_with_caller_fn p_buf_ptr = nullptr;
mpp_frame_init_fn p_frame_init = nullptr;
mpp_frame_deinit_fn p_frame_deinit = nullptr;
mpp_frame_set_width_fn p_frame_w = nullptr;
mpp_frame_set_height_fn p_frame_h = nullptr;
mpp_frame_set_hor_stride_fn p_frame_hs = nullptr;
mpp_frame_set_ver_stride_fn p_frame_vs = nullptr;
mpp_frame_set_fmt_fn p_frame_fmt = nullptr;
mpp_frame_set_buffer_fn p_frame_buf = nullptr;
mpp_frame_set_pts_fn p_frame_pts = nullptr;
mpp_packet_get_data_fn p_pkt_data = nullptr;
mpp_packet_get_length_fn p_pkt_len = nullptr;
mpp_packet_get_pts_fn p_pkt_pts = nullptr;
mpp_packet_get_eos_fn p_pkt_eos = nullptr;
mpp_packet_deinit_fn p_pkt_deinit = nullptr;

bool load_mpp_lib() {
    static bool tried = false;
    static bool ok = false;
    if (tried) return ok;
    tried = true;
    void* h = dlopen("librockchip_mpp.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("librockchip_mpp.so.1", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        LOG_WARN_FMT("[MppVideoEncoder] dlopen librockchip_mpp failed: {}", dlerror());
        return false;
    }
    auto sym = [&](const char* n) { return dlsym(h, n); };
    p_create = reinterpret_cast<mpp_create_fn>(sym("mpp_create"));
    p_init = reinterpret_cast<mpp_init_fn>(sym("mpp_init"));
    p_destroy = reinterpret_cast<mpp_destroy_fn>(sym("mpp_destroy"));
    p_cfg_init = reinterpret_cast<mpp_enc_cfg_init_fn>(sym("mpp_enc_cfg_init"));
    p_cfg_deinit = reinterpret_cast<mpp_enc_cfg_deinit_fn>(sym("mpp_enc_cfg_deinit"));
    p_cfg_s32 = reinterpret_cast<mpp_enc_cfg_set_s32_fn>(sym("mpp_enc_cfg_set_s32"));
    p_cfg_u32 = reinterpret_cast<mpp_enc_cfg_set_u32_fn>(sym("mpp_enc_cfg_set_u32"));
    p_buf_group_get = reinterpret_cast<mpp_buffer_group_get_fn>(sym("mpp_buffer_group_get"));
    p_buf_group_put = reinterpret_cast<mpp_buffer_group_put_fn>(sym("mpp_buffer_group_put"));
    p_buf_get = reinterpret_cast<mpp_buffer_get_with_tag_fn>(sym("mpp_buffer_get_with_tag"));
    p_buf_put = reinterpret_cast<mpp_buffer_put_fn>(sym("mpp_buffer_put"));
    p_buf_ptr = reinterpret_cast<mpp_buffer_get_ptr_with_caller_fn>(sym("mpp_buffer_get_ptr_with_caller"));
    p_frame_init = reinterpret_cast<mpp_frame_init_fn>(sym("mpp_frame_init"));
    p_frame_deinit = reinterpret_cast<mpp_frame_deinit_fn>(sym("mpp_frame_deinit"));
    p_frame_w = reinterpret_cast<mpp_frame_set_width_fn>(sym("mpp_frame_set_width"));
    p_frame_h = reinterpret_cast<mpp_frame_set_height_fn>(sym("mpp_frame_set_height"));
    p_frame_hs = reinterpret_cast<mpp_frame_set_hor_stride_fn>(sym("mpp_frame_set_hor_stride"));
    p_frame_vs = reinterpret_cast<mpp_frame_set_ver_stride_fn>(sym("mpp_frame_set_ver_stride"));
    p_frame_fmt = reinterpret_cast<mpp_frame_set_fmt_fn>(sym("mpp_frame_set_fmt"));
    p_frame_buf = reinterpret_cast<mpp_frame_set_buffer_fn>(sym("mpp_frame_set_buffer"));
    p_frame_pts = reinterpret_cast<mpp_frame_set_pts_fn>(sym("mpp_frame_set_pts"));
    p_pkt_data = reinterpret_cast<mpp_packet_get_data_fn>(sym("mpp_packet_get_data"));
    p_pkt_len = reinterpret_cast<mpp_packet_get_length_fn>(sym("mpp_packet_get_length"));
    p_pkt_pts = reinterpret_cast<mpp_packet_get_pts_fn>(sym("mpp_packet_get_pts"));
    p_pkt_eos = reinterpret_cast<mpp_packet_get_eos_fn>(sym("mpp_packet_get_eos"));
    p_pkt_deinit = reinterpret_cast<mpp_packet_deinit_fn>(sym("mpp_packet_deinit"));
    ok = p_create && p_init && p_destroy && p_cfg_init && p_cfg_deinit &&
         p_cfg_s32 && p_cfg_u32 && p_buf_group_get && p_buf_group_put &&
         p_buf_get && p_buf_put && p_buf_ptr && p_frame_init && p_frame_deinit &&
         p_frame_w && p_frame_h && p_frame_hs && p_frame_vs && p_frame_fmt &&
         p_frame_buf && p_frame_pts && p_pkt_data && p_pkt_len && p_pkt_eos &&
         p_pkt_deinit;
    if (!ok) LOG_ERROR("[MppVideoEncoder] Failed to load MPP API symbols");
    return ok;
}

} // namespace

MppVideoEncoder::MppVideoEncoder() = default;

MppVideoEncoder::~MppVideoEncoder() {
    close();
}

bool MppVideoEncoder::isAvailable() const {
    return load_mpp_lib();
}

bool MppVideoEncoder::open(const VideoEncoderConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (opened_) return true;
    width_ = config.width;
    height_ = config.height;
    gop_ = config.gop > 0 ? config.gop : config.fps;

    frame_size_ = static_cast<size_t>(width_) * height_ * 3 / 2;

    if (!initMpp(config)) return false;
    if (!fetchHeaderSync()) return false;

    opened_ = true;
    LOG_INFO_FMT("[MppVideoEncoder] Opened: {}x{} @ {}kbps gop={}",
                 width_, height_, config.bitrate_kbps, gop_);
    return true;
}

bool MppVideoEncoder::initMpp(const VideoEncoderConfig& config) {
    if (!load_mpp_lib()) return false;

    MPP_RET ret = p_create(&ctx_, &mpi_);
    if (ret != MPP_OK || !mpi_) {
        LOG_ERROR_FMT("[MppVideoEncoder] mpp_create failed: {}", static_cast<int>(ret));
        return false;
    }
    ret = p_init(ctx_, MPP_CTX_ENC, MPP_VIDEO_CodingAVC);
    if (ret != MPP_OK) {
        LOG_ERROR_FMT("[MppVideoEncoder] mpp_init ENC/AVC failed: {}", static_cast<int>(ret));
        return false;
    }

    ret = p_cfg_init(&cfg_);
    if (ret != MPP_OK) {
        LOG_ERROR("[MppVideoEncoder] mpp_enc_cfg_init failed");
        return false;
    }
    ret = mpi_->control(ctx_, MPP_ENC_GET_CFG, cfg_);
    if (ret != MPP_OK) {
        LOG_WARN("[MppVideoEncoder] GET_CFG failed, using defaults");
    }

    const RK_U32 hor_stride = MPP_ALIGN(width_, 16);
    const RK_U32 ver_stride = MPP_ALIGN(height_, 16);

    p_cfg_s32(cfg_, "prep:width", width_);
    p_cfg_s32(cfg_, "prep:height", height_);
    p_cfg_s32(cfg_, "prep:hor_stride", static_cast<RK_S32>(hor_stride));
    p_cfg_s32(cfg_, "prep:ver_stride", static_cast<RK_S32>(ver_stride));
    p_cfg_s32(cfg_, "prep:format", static_cast<RK_S32>(MPP_FMT_YUV420P));
    p_cfg_s32(cfg_, "rc:mode", static_cast<RK_S32>(MPP_ENC_RC_MODE_CBR));
    p_cfg_s32(cfg_, "rc:fps_in_flex", 0);
    p_cfg_s32(cfg_, "rc:fps_in_num", config.fps);
    p_cfg_s32(cfg_, "rc:fps_in_denorm", 1);
    p_cfg_s32(cfg_, "rc:fps_out_flex", 0);
    p_cfg_s32(cfg_, "rc:fps_out_num", config.fps);
    p_cfg_s32(cfg_, "rc:fps_out_denorm", 1);
    p_cfg_s32(cfg_, "rc:gop", gop_);
    p_cfg_u32(cfg_, "rc:bps_target", static_cast<RK_U32>(config.bitrate_kbps * 1000));
    p_cfg_u32(cfg_, "rc:bps_max", static_cast<RK_U32>(config.bitrate_kbps * 1500));
    p_cfg_u32(cfg_, "rc:bps_min", static_cast<RK_U32>(config.bitrate_kbps / 2 * 1000));
    p_cfg_s32(cfg_, "h264:profile", 100);  // high
    p_cfg_s32(cfg_, "h264:level", 41);
    p_cfg_s32(cfg_, "h264:cabac_en", 1);
    p_cfg_s32(cfg_, "h264:cabac_idc", 0);
    p_cfg_s32(cfg_, "h264:trans8x8", 1);

    ret = mpi_->control(ctx_, MPP_ENC_SET_CFG, cfg_);
    if (ret != MPP_OK) {
        LOG_ERROR_FMT("[MppVideoEncoder] MPP_ENC_SET_CFG failed: {}", static_cast<int>(ret));
        return false;
    }

    ret = p_buf_group_get(&buf_group_, MPP_BUFFER_TYPE_ION, MPP_BUFFER_INTERNAL,
                          MODULE_TAG, __FUNCTION__);
    if (ret != MPP_OK) {
        ret = p_buf_group_get(&buf_group_, MPP_BUFFER_TYPE_DRM, MPP_BUFFER_INTERNAL,
                              MODULE_TAG, __FUNCTION__);
    }
    if (ret != MPP_OK) {
        LOG_ERROR("[MppVideoEncoder] mpp_buffer_group_get failed");
        return false;
    }

    ret = p_buf_get(buf_group_, &frame_buf_, frame_size_, MODULE_TAG, __FUNCTION__);
    if (ret != MPP_OK) {
        LOG_ERROR("[MppVideoEncoder] mpp_buffer_get failed");
        return false;
    }

    ret = p_frame_init(&frame_);
    if (ret != MPP_OK) {
        LOG_ERROR("[MppVideoEncoder] mpp_frame_init failed");
        return false;
    }
    p_frame_w(frame_, width_);
    p_frame_h(frame_, height_);
    p_frame_hs(frame_, hor_stride);
    p_frame_vs(frame_, ver_stride);
    p_frame_fmt(frame_, MPP_FMT_YUV420P);
    p_frame_buf(frame_, frame_buf_);

    return true;
}

bool MppVideoEncoder::fetchHeaderSync() {
    MppPacket hdr = nullptr;
    MPP_RET ret = mpi_->control(ctx_, MPP_ENC_GET_HDR_SYNC, &hdr);
    if (ret != MPP_OK || !hdr) {
        LOG_ERROR_FMT("[MppVideoEncoder] MPP_ENC_GET_HDR_SYNC failed: {}", static_cast<int>(ret));
        return false;
    }
    const uint8_t* data = static_cast<const uint8_t*>(p_pkt_data(hdr));
    const size_t size = p_pkt_len(hdr);
    if (!data || size == 0) {
        LOG_ERROR("[MppVideoEncoder] empty header sync");
        p_pkt_deinit(&hdr);
        return false;
    }
    // 契约：extradata 为 AnnexB（SPS/PPS）；AVCDecoderConfigurationRecord
    // 由 muxer（ff_isom_write_avcc）生成，与 ffmpeg 软编路径行为一致
    extradata_.assign(data, data + size);
    p_pkt_deinit(&hdr);
    return true;
}

bool MppVideoEncoder::encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                             std::vector<EncodedPacket>& packets) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_ || !yuv420p || size < frame_size_) {
        LOG_ERROR_FMT("[MppVideoEncoder] encode invalid (opened={}, size={})",
                      opened_, size);
        return false;
    }

    // 拷入专用编码缓冲（同步 encode 接口下安全）
    uint8_t* dst = static_cast<uint8_t*>(p_buf_ptr(frame_buf_, __FUNCTION__));
    if (!dst) return false;
    memcpy(dst, yuv420p, frame_size_);
    p_frame_pts(frame_, pts);

    MppPacket pkt = nullptr;
    MPP_RET ret = mpi_->encode(ctx_, frame_, &pkt);
    if (ret != MPP_OK || !pkt) {
        LOG_ERROR_FMT("[MppVideoEncoder] mpi->encode failed: {}", static_cast<int>(ret));
        return false;
    }

    const void* data = p_pkt_data(pkt);
    const size_t len = p_pkt_len(pkt);
    if (data && len > 0 && !p_pkt_eos(pkt)) {
        EncodedPacket out;
        out.data = static_cast<const uint8_t*>(data);
        out.size = len;
        out.pts = pts;
        out.dts = pts;
        // AnnexB 首个 NAL 类型 5 = IDR 关键帧
        const uint8_t* d = out.data;
        if (out.size > 5 &&
            ((d[0] == 0 && d[1] == 0 && d[2] == 0 && d[3] == 1) ||
             (d[0] == 0 && d[1] == 0 && d[2] == 1))) {
            const size_t sc = (d[0] == 0 && d[1] == 0 && d[2] == 0) ? 4 : 3;
            out.keyframe = ((d[sc] & 0x1F) == 5);
        }
        packets.push_back(out);
    }
    p_pkt_deinit(&pkt);
    return true;
}

void MppVideoEncoder::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!opened_ && !ctx_) return;

    if (frame_) p_frame_deinit(&frame_);
    frame_ = nullptr;
    if (frame_buf_) p_buf_put(frame_buf_);
    frame_buf_ = nullptr;
    if (buf_group_) p_buf_group_put(buf_group_);
    buf_group_ = nullptr;
    if (cfg_) p_cfg_deinit(cfg_);
    cfg_ = nullptr;
    if (ctx_) p_destroy(ctx_);
    ctx_ = nullptr;
    mpi_ = nullptr;
    opened_ = false;
    LOG_INFO("[MppVideoEncoder] Closed");
}

// 注册到工厂（始终注册；dlopen 失败时 create 后 isAvailable 为 false，
// 节点据此回退软件编码）
REGISTER_VIDEO_ENCODER("mpp_h264", MppVideoEncoder)

} // namespace hal
} // namespace ai_stream
