// src/hal/mpp/mpp_video_codec.cpp — Rockchip MPP 硬解（dlopen，x86 仅头文件编译，板端真解码）
#include "mpp_video_codec.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <dlfcn.h>
#include <unistd.h>
#include <cstring>
#include <mutex>

#ifdef WITH_RKNN
#include "rockchip/rk_mpi.h"
#include "rockchip/mpp_frame.h"
#include "rockchip/mpp_buffer.h"
#include "rockchip/mpp_packet.h"
#endif

namespace ai_stream {
namespace hal {

#ifdef WITH_RKNN
static void* g_mpp_handle = nullptr;
static bool g_mpp_tried = false;
static bool g_mpp_loaded = false;

// 关键符号 typedef
typedef MPP_RET (*mpp_create_fn)(MppCtx*, MppApi**);
typedef MPP_RET (*mpp_init_fn)(MppCtx, MppCtxType, MppCodingType);
typedef MPP_RET (*mpp_destroy_fn)(MppCtx);
typedef MPP_RET (*mpp_packet_init_fn)(MppPacket*, void*, size_t);
typedef MPP_RET (*mpp_packet_deinit_fn)(MppPacket*);
// mpp_buffer_group_get: (group, type, mode, tag, func) — header 中 mpp_buffer_group_get_internal 是宏
typedef MPP_RET (*mpp_buffer_group_get_fn)(MppBufferGroup*, MppBufferType, int mode, const char* tag, const char* func);
#define MPP_BUF_MODE_INTERNAL 0
typedef MPP_RET (*mpp_buffer_group_put_fn)(MppBufferGroup);
typedef MPP_RET (*mpp_frame_deinit_fn)(MppFrame*);
typedef MPP_RET (*mpp_dec_cfg_init_fn)(MppDecCfg*);
typedef MPP_RET (*mpp_dec_cfg_deinit_fn)(MppDecCfg);
typedef MPP_RET (*mpp_dec_cfg_set_u32_fn)(MppDecCfg, const char*, RK_U32);
typedef RK_U32 (*mpp_frame_get_width_fn)(MppFrame);
typedef RK_U32 (*mpp_frame_get_height_fn)(MppFrame);
typedef RK_U32 (*mpp_frame_get_hor_stride_fn)(MppFrame);
typedef RK_U32 (*mpp_frame_get_ver_stride_fn)(MppFrame);
typedef size_t (*mpp_frame_get_buf_size_fn)(MppFrame);
typedef RK_U32 (*mpp_frame_get_info_change_fn)(MppFrame);
typedef MppBuffer (*mpp_frame_get_buffer_fn)(MppFrame);
typedef void* (*mpp_buffer_get_ptr_with_caller_fn)(MppBuffer, const char*);

static mpp_create_fn p_mpp_create = nullptr;
static mpp_init_fn p_mpp_init = nullptr;
static mpp_destroy_fn p_mpp_destroy = nullptr;
static mpp_packet_init_fn p_mpp_packet_init = nullptr;
static mpp_packet_deinit_fn p_mpp_packet_deinit = nullptr;

static bool load_mpp() {
    if (g_mpp_tried) return g_mpp_loaded;
    g_mpp_tried = true;
    const char* cand[] = {"3rd_party/rk_platform/mpp/lib/aarch64/librockchip_mpp.so","librockchip_mpp.so", nullptr};
    for (int i=0;cand[i];++i){ g_mpp_handle = dlopen(cand[i], RTLD_NOW); if(g_mpp_handle) break; }
    if(!g_mpp_handle){ LOG_DEBUG_FMT("[MppVideoCodec] dlopen librockchip_mpp.so fail: {}", dlerror()?dlerror():"unknown"); return false; }
    p_mpp_create = (mpp_create_fn)dlsym(g_mpp_handle,"mpp_create");
    p_mpp_init = (mpp_init_fn)dlsym(g_mpp_handle,"mpp_init");
    p_mpp_destroy = (mpp_destroy_fn)dlsym(g_mpp_handle,"mpp_destroy");
    p_mpp_packet_init = (mpp_packet_init_fn)dlsym(g_mpp_handle,"mpp_packet_init");
    p_mpp_packet_deinit = (mpp_packet_deinit_fn)dlsym(g_mpp_handle,"mpp_packet_deinit");
    if(!p_mpp_create || !p_mpp_init){ LOG_WARN("[MppVideoCodec] mpp symbols missing"); dlclose(g_mpp_handle); g_mpp_handle=nullptr; return false; }
    g_mpp_loaded = true;
    LOG_INFO("[MppVideoCodec] librockchip_mpp.so loaded");
    return true;
}
#endif

struct MppPriv {
#ifdef WITH_RKNN
    MppCtx ctx = nullptr;
    MppApi* api = nullptr;
    MppBufferGroup frm_grp = nullptr;
    MppCodingType coding = MPP_VIDEO_CodingAVC;
    bool info_changed = false;
#endif
    std::mutex mtx;
};

MppVideoCodec::MppVideoCodec(): mpp_ctx_(nullptr), mpp_api_(nullptr) {
    mpp_ctx_ = new MppPriv();
    initialized_ = initMpp();
    if (initialized_) LOG_DEBUG("[MppVideoCodec] Initialized");
}
MppVideoCodec::~MppVideoCodec(){ cleanup(); if(mpp_ctx_){ delete static_cast<MppPriv*>(mpp_ctx_); mpp_ctx_=nullptr; } }

void MppVideoCodec::release(){ cleanup(); LOG_DEBUG("[MppVideoCodec] Released"); }

bool MppVideoCodec::isAvailable() const {
#ifdef WITH_RKNN
    if (initialized_) return true;
    if (access("/dev/mpp_service", F_OK)==0 || access("/dev/rkvdec", F_OK)==0) return true;
    return false;
#else
    return false;
#endif
}

bool MppVideoCodec::initMpp() {
#ifdef WITH_RKNN
    if (!load_mpp()){ LOG_DEBUG("[MppVideoCodec] MPP not available on this host, fallback to FFmpeg"); return false; }
    auto priv = static_cast<MppPriv*>(mpp_ctx_);
    if (codec_name_.empty()) return true;
    MPP_RET ret;
    MppCodingType type = MPP_VIDEO_CodingAVC;
    if (codec_name_=="h265"||codec_name_=="hevc") type = MPP_VIDEO_CodingHEVC;
    else if (codec_name_=="h264"||codec_name_=="avc") type = MPP_VIDEO_CodingAVC;
    else { LOG_WARN_FMT("[MppVideoCodec] unsupported codec {}, default h264", codec_name_); type = MPP_VIDEO_CodingAVC; }
    priv->coding = type;
    ret = p_mpp_create(&priv->ctx, &priv->api);
    if(ret!=MPP_OK){ LOG_ERROR_FMT("[MppVideoCodec] mpp_create fail {}", static_cast<int>(ret)); return false; }
    ret = p_mpp_init(priv->ctx, MPP_CTX_DEC, type);
    if(ret!=MPP_OK){ LOG_ERROR_FMT("[MppVideoCodec] mpp_init fail {}", static_cast<int>(ret)); p_mpp_destroy(priv->ctx); priv->ctx=nullptr; return false; }
    // 配置 split_parse
    MppDecCfg cfg=nullptr;
    { auto fn=(mpp_dec_cfg_init_fn)dlsym(g_mpp_handle,"mpp_dec_cfg_init"); if(fn) fn(&cfg); }
    ret = priv->api->control(priv->ctx, MPP_DEC_GET_CFG, cfg);
    if(ret==MPP_OK){
        auto fnSet=(mpp_dec_cfg_set_u32_fn)dlsym(g_mpp_handle,"mpp_dec_cfg_set_u32");
        if(fnSet) fnSet(cfg,"base:split_parse",1);
        priv->api->control(priv->ctx, MPP_DEC_SET_CFG, cfg);
    }
    { auto fn=(mpp_dec_cfg_deinit_fn)dlsym(g_mpp_handle,"mpp_dec_cfg_deinit"); if(fn) fn(cfg); }
    mpp_api_ = priv->api;
    LOG_INFO_FMT("[MppVideoCodec] MPP decoder created codec={}", codec_name_);
    return true;
#else
    LOG_DEBUG("[MppVideoCodec] WITH_RKNN not enabled");
    return false;
#endif
}

void MppVideoCodec::cleanup(){
#ifdef WITH_RKNN
    auto priv = mpp_ctx_ ? static_cast<MppPriv*>(mpp_ctx_) : nullptr;
    if(priv){
        std::lock_guard<std::mutex> lk(priv->mtx);
        if(priv->ctx && p_mpp_destroy) { p_mpp_destroy(priv->ctx); priv->ctx=nullptr; priv->api=nullptr; }
        if(priv->frm_grp){ auto fn = (MPP_RET(*)(MppBufferGroup))dlsym(g_mpp_handle,"mpp_buffer_group_put"); if(fn) fn(priv->frm_grp); priv->frm_grp=nullptr; }
    }
#endif
    initialized_=false; mpp_api_=nullptr;
}

static void handle_info_change(MppPriv* priv, MppFrame mpp_frame){
#ifdef WITH_RKNN
    auto fnW=(mpp_frame_get_width_fn)dlsym(g_mpp_handle,"mpp_frame_get_width");
    auto fnH=(mpp_frame_get_height_fn)dlsym(g_mpp_handle,"mpp_frame_get_height");
    auto fnHS=(mpp_frame_get_hor_stride_fn)dlsym(g_mpp_handle,"mpp_frame_get_hor_stride");
    auto fnVS=(mpp_frame_get_ver_stride_fn)dlsym(g_mpp_handle,"mpp_frame_get_ver_stride");
    auto fnBS=(mpp_frame_get_buf_size_fn)dlsym(g_mpp_handle,"mpp_frame_get_buf_size");
    RK_U32 w=fnW?fnW(mpp_frame):0, h=fnH?fnH(mpp_frame):0;
    RK_U32 hs=fnHS?fnHS(mpp_frame):0, vs=fnVS?fnVS(mpp_frame):0;
    size_t buf_size=fnBS?fnBS(mpp_frame):0;
    LOG_INFO_FMT("[MppVideoCodec] info change {}x{} stride {}x{} buf {}", w, h, hs, vs, buf_size);
    if(!priv->frm_grp){
        auto fnGet=(mpp_buffer_group_get_fn)dlsym(g_mpp_handle,"mpp_buffer_group_get");
        if(fnGet) fnGet(&priv->frm_grp, MPP_BUFFER_TYPE_DRM, MPP_BUF_MODE_INTERNAL, "mpp", "decode");
        if(!priv->frm_grp && fnGet) fnGet(&priv->frm_grp, MPP_BUFFER_TYPE_DMA_HEAP, MPP_BUF_MODE_INTERNAL, "mpp", "decode");
        if(priv->frm_grp) priv->api->control(priv->ctx, MPP_DEC_SET_EXT_BUF_GROUP, priv->frm_grp);
    }
    if(priv->frm_grp){
        auto fnLimit=(MPP_RET(*)(MppBufferGroup,size_t,RK_U32))dlsym(g_mpp_handle,"mpp_buffer_group_limit_config");
        if(fnLimit) fnLimit(priv->frm_grp, buf_size, 24);
    }
    priv->api->control(priv->ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
#endif
}

bool MppVideoCodec::init(const std::string& codec_name, const uint8_t* extradata, int extradata_size){
    codec_name_=codec_name; extradata_=extradata; extradata_size_=extradata_size;
    LOG_INFO_FMT("[MppVideoCodec] init codec={} extradata={}", codec_name, extradata_size);
#ifdef WITH_RKNN
    auto priv = static_cast<MppPriv*>(mpp_ctx_);
    if(!priv->ctx){
        if(!initMpp()) return false;
        initialized_=true;
    }
    return initialized_;
#else
    return false;
#endif
}

bool MppVideoCodec::decode(const uint8_t* packet_data, int packet_size, DecodedFrame& frame){
    if(!initialized_ || !packet_data || packet_size<=0) return false;
#ifdef WITH_RKNN
    auto priv = static_cast<MppPriv*>(mpp_ctx_);
    if(!priv || !priv->ctx || !priv->api) return false;
    std::lock_guard<std::mutex> lk(priv->mtx);

    MppPacket pkt=nullptr;
    if(p_mpp_packet_init) p_mpp_packet_init(&pkt, const_cast<uint8_t*>(packet_data), packet_size);
    if(!pkt) return false;

    MPP_RET ret = priv->api->decode_put_packet(priv->ctx, pkt);
    if(ret!=MPP_OK){
        if(ret == -1012){ // MPP_ERR_BUFFER_FULL: drain to free buffer, then retry
            for(int drain=0;drain<3;++drain){
                MppFrame drain_frame=nullptr;
                priv->api->decode_get_frame(priv->ctx, &drain_frame);
                if(drain_frame){
                    auto fnIC=(mpp_frame_get_info_change_fn)dlsym(g_mpp_handle,"mpp_frame_get_info_change");
                    if(fnIC && fnIC(drain_frame)) handle_info_change(priv, drain_frame);
                    auto fnDI=(MPP_RET(*)(MppFrame*))dlsym(g_mpp_handle,"mpp_frame_deinit");
                    if(fnDI) fnDI(&drain_frame);
                }
                usleep(2000);
            }
            ret = priv->api->decode_put_packet(priv->ctx, pkt);
        }
        if(ret!=MPP_OK){ if(p_mpp_packet_deinit) p_mpp_packet_deinit(&pkt); return false; }
    }

    // Get decoded frame
    MppFrame mpp_frame=nullptr;
    for(int i=0;i<20;++i){
        ret = priv->api->decode_get_frame(priv->ctx, &mpp_frame);
        if(ret==MPP_ERR_TIMEOUT){ usleep(5000); continue; }
        if(ret!=MPP_OK) break;
        if(mpp_frame) break;
    }
    if(p_mpp_packet_deinit) p_mpp_packet_deinit(&pkt);
    if(!mpp_frame) return false;

    // Handle info change
    auto fnIC=(mpp_frame_get_info_change_fn)dlsym(g_mpp_handle,"mpp_frame_get_info_change");
    if(fnIC && fnIC(mpp_frame)){
        handle_info_change(priv, mpp_frame);
        auto fnDI=(MPP_RET(*)(MppFrame*))dlsym(g_mpp_handle,"mpp_frame_deinit");
        if(fnDI) fnDI(&mpp_frame);
        return false;
    }

    // Normal frame — extract data
    auto fnW=(mpp_frame_get_width_fn)dlsym(g_mpp_handle,"mpp_frame_get_width");
    auto fnH=(mpp_frame_get_height_fn)dlsym(g_mpp_handle,"mpp_frame_get_height");
    auto fnHS=(mpp_frame_get_hor_stride_fn)dlsym(g_mpp_handle,"mpp_frame_get_hor_stride");
    auto fnVS=(mpp_frame_get_ver_stride_fn)dlsym(g_mpp_handle,"mpp_frame_get_ver_stride");
    RK_U32 w=fnW?fnW(mpp_frame):0, h=fnH?fnH(mpp_frame):0;
    RK_U32 hs=fnHS?fnHS(mpp_frame):0, vs=fnVS?fnVS(mpp_frame):0;

    auto fnBuf=(mpp_frame_get_buffer_fn)dlsym(g_mpp_handle,"mpp_frame_get_buffer");
    MppBuffer buf=fnBuf?fnBuf(mpp_frame):nullptr;
    void* vir=nullptr;
    if(buf){
        auto fnPtr=(mpp_buffer_get_ptr_with_caller_fn)dlsym(g_mpp_handle,"mpp_buffer_get_ptr_with_caller");
        if(fnPtr) vir=fnPtr(buf, "decode");
    }
    if(!vir){
        auto fnDI=(MPP_RET(*)(MppFrame*))dlsym(g_mpp_handle,"mpp_frame_deinit");
        if(fnDI) fnDI(&mpp_frame);
        return false;
    }

    // Copy NV12: Y + UV
    size_t y_size = hs * vs;
    size_t uv_size = hs * vs / 2;
    size_t total = y_size + uv_size;
    uint8_t* out = new uint8_t[total];
    memcpy(out, vir, total);

    frame.data = out; frame.data_uv = out + y_size;
    frame.width = w; frame.height = h; frame.pitch = hs; frame.pitch_uv = hs;
    frame.format = 23; // AV_PIX_FMT_NV12
    frame.owns_data = true;

    auto fnDI=(MPP_RET(*)(MppFrame*))dlsym(g_mpp_handle,"mpp_frame_deinit");
    if(fnDI) fnDI(&mpp_frame);
    return true;
#else
    (void)frame;
    return false;
#endif
}

#ifdef WITH_RKNN
REGISTER_VIDEO_CODEC(VideoCodecBackend::MPP, MppVideoCodec)
#endif
} // namespace hal
} // namespace ai_stream
