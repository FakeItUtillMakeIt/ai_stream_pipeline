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

// 关键符号
typedef MPP_RET (*mpp_create_fn)(MppCtx*, MppApi**);
typedef MPP_RET (*mpp_init_fn)(MppCtx, MppCtxType, MppCodingType);
typedef MPP_RET (*mpp_destroy_fn)(MppCtx);
typedef MPP_RET (*mpp_packet_init_fn)(MppPacket*, void*, size_t);
typedef MPP_RET (*mpp_packet_deinit_fn)(MppPacket*);
typedef void (*mpp_packet_set_data_fn)(MppPacket, void*);
typedef void (*mpp_packet_set_size_fn)(MppPacket, size_t);
typedef void (*mpp_packet_set_pos_fn)(MppPacket, void*);
typedef void (*mpp_packet_set_length_fn)(MppPacket, size_t);
typedef void (*mpp_packet_set_pts_fn)(MppPacket, RK_S64);
typedef void (*mpp_packet_set_eos_fn)(MppPacket);
typedef MPP_RET (*mpp_buffer_group_get_internal_fn)(MppBufferGroup*, MppBufferType);
typedef MPP_RET (*mpp_buffer_group_put_fn)(MppBufferGroup);
typedef MPP_RET (*mpp_frame_deinit_fn)(MppFrame*);
typedef MPP_RET (*mpp_dec_cfg_init_fn)(MppDecCfg*);
typedef MPP_RET (*mpp_dec_cfg_deinit_fn)(MppDecCfg);
typedef MPP_RET (*mpp_dec_cfg_set_u32_fn)(MppDecCfg, const char*, RK_U32);
typedef void (*mpp_packet_set_data_fn2)(MppPacket, void*);
typedef void (*mpp_packet_set_size_fn2)(MppPacket, size_t);
typedef void (*mpp_packet_set_pos_fn2)(MppPacket, void*);
typedef void (*mpp_packet_set_length_fn2)(MppPacket, size_t);
typedef RK_U32 (*mpp_frame_get_width_fn)(MppFrame);
typedef RK_U32 (*mpp_frame_get_height_fn)(MppFrame);
typedef RK_U32 (*mpp_frame_get_hor_stride_fn)(MppFrame);
typedef RK_U32 (*mpp_frame_get_ver_stride_fn)(MppFrame);
typedef size_t (*mpp_frame_get_buf_size_fn)(MppFrame);
typedef RK_U32 (*mpp_frame_get_info_change_fn)(MppFrame);
typedef MppBuffer (*mpp_frame_get_buffer_fn)(MppFrame);
typedef void* (*mpp_buffer_get_ptr_fn)(MppBuffer);

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
    bool need_split = true;
#endif
    std::mutex mtx;
};

MppVideoCodec::MppVideoCodec(): mpp_ctx_(nullptr), mpp_api_(nullptr) {
    // mpp_priv_ 用 void* 持有，避免头文件依赖
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
    // codec_name_ 可能尚未设置，延迟到 init() 再真正 mpp_init；此处仅检查库
    // 若已有 codec_name_ 则立即初始化
    if (codec_name_.empty()) return true; // 延迟初始化
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
    if(ret==MPP_OK){ { auto fn=(mpp_dec_cfg_set_u32_fn)dlsym(g_mpp_handle,"mpp_dec_cfg_set_u32"); if(fn) fn(cfg,"base:split_parse",1); } priv->api->control(priv->ctx, MPP_DEC_SET_CFG, cfg); }
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

bool MppVideoCodec::init(const std::string& codec_name, const uint8_t* extradata, int extradata_size){
    codec_name_=codec_name; extradata_=extradata; extradata_size_=extradata_size;
    LOG_INFO_FMT("[MppVideoCodec] init codec={} extradata={}", codec_name, extradata_size);
#ifdef WITH_RKNN
    // 若之前未初始化（codec_name 为空时延迟），现在真正创建
    auto priv = static_cast<MppPriv*>(mpp_ctx_);
    if(!priv->ctx){
        if(!initMpp()) return false;
        initialized_=true;
    }
    // 注入 extradata (SPS/PPS) 作为首包
    if(extradata && extradata_size>0 && priv->api && priv->ctx){
        MppPacket pkt=nullptr;
        if(p_mpp_packet_init) p_mpp_packet_init(&pkt, const_cast<uint8_t*>(extradata), extradata_size);
        if(pkt){
            { auto fn=(mpp_packet_set_data_fn2)dlsym(g_mpp_handle,"mpp_packet_set_data"); if(fn) fn(pkt, const_cast<uint8_t*>(extradata)); }
            { auto fn=(mpp_packet_set_size_fn2)dlsym(g_mpp_handle,"mpp_packet_set_size"); if(fn) fn(pkt, extradata_size); }
            { auto fn=(mpp_packet_set_pos_fn2)dlsym(g_mpp_handle,"mpp_packet_set_pos"); if(fn) fn(pkt, const_cast<uint8_t*>(extradata)); }
            { auto fn=(mpp_packet_set_length_fn2)dlsym(g_mpp_handle,"mpp_packet_set_length"); if(fn) fn(pkt, extradata_size); }
            priv->api->decode_put_packet(priv->ctx, pkt);
            // 不等待帧，extradata 仅配置解码器
            if(p_mpp_packet_deinit) p_mpp_packet_deinit(&pkt);
        }
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
    MPP_RET ret;
    if(p_mpp_packet_init) p_mpp_packet_init(&pkt, nullptr, 0);
    if(!pkt) return false;
    // 绑定外部内存（零拷贝）
    { auto fn=(mpp_packet_set_data_fn2)dlsym(g_mpp_handle,"mpp_packet_set_data"); if(fn) fn(pkt, const_cast<uint8_t*>(packet_data)); }
    { auto fn=(mpp_packet_set_size_fn2)dlsym(g_mpp_handle,"mpp_packet_set_size"); if(fn) fn(pkt, packet_size); }
    { auto fn=(mpp_packet_set_pos_fn2)dlsym(g_mpp_handle,"mpp_packet_set_pos"); if(fn) fn(pkt, const_cast<uint8_t*>(packet_data)); }
    { auto fn=(mpp_packet_set_length_fn2)dlsym(g_mpp_handle,"mpp_packet_set_length"); if(fn) fn(pkt, packet_size); }
    ret = priv->api->decode_put_packet(priv->ctx, pkt);
    if(ret!=MPP_OK){ LOG_DEBUG_FMT("[MppVideoCodec] decode_put_packet ret {}", static_cast<int>(ret)); if(p_mpp_packet_deinit) p_mpp_packet_deinit(&pkt); return false; }
    // 尝试取帧（最多 5 次超时重试）
    MppFrame mpp_frame=nullptr;
    for(int i=0;i<5;++i){
        ret = priv->api->decode_get_frame(priv->ctx, &mpp_frame);
        if(ret==MPP_ERR_TIMEOUT){ usleep(2000); continue; }
        if(ret!=MPP_OK) break;
        if(mpp_frame) break;
    }
    if(p_mpp_packet_deinit) p_mpp_packet_deinit(&pkt);
    if(!mpp_frame) return false;
    // info change 处理
    if(({ auto fn=(mpp_frame_get_info_change_fn)dlsym(g_mpp_handle,"mpp_frame_get_info_change"); fn?fn(mpp_frame):0; })){
        RK_U32 w = ({ auto fn=(mpp_frame_get_width_fn)dlsym(g_mpp_handle,"mpp_frame_get_width"); fn?fn(mpp_frame):0; }), h = ({ auto fn=(mpp_frame_get_height_fn)dlsym(g_mpp_handle,"mpp_frame_get_height"); fn?fn(mpp_frame):0; });
        RK_U32 hs = ({ auto fn=(mpp_frame_get_hor_stride_fn)dlsym(g_mpp_handle,"mpp_frame_get_hor_stride"); fn?fn(mpp_frame):0; }), vs = ({ auto fn=(mpp_frame_get_ver_stride_fn)dlsym(g_mpp_handle,"mpp_frame_get_ver_stride"); fn?fn(mpp_frame):0; });
        size_t buf_size = ({ auto fn=(mpp_frame_get_buf_size_fn)dlsym(g_mpp_handle,"mpp_frame_get_buf_size"); fn?fn(mpp_frame):0; });
        LOG_INFO_FMT("[MppVideoCodec] info change w={} h={} stride {}x{} buf_size {}", w,h,hs,vs,buf_size);
        if(!priv->frm_grp){
            auto fnGet = (MPP_RET(*)(MppBufferGroup*,MppBufferType))dlsym(g_mpp_handle,"mpp_buffer_group_get_internal");
            if(fnGet) fnGet(&priv->frm_grp, MPP_BUFFER_TYPE_DRM);
            if(priv->frm_grp) priv->api->control(priv->ctx, MPP_DEC_SET_EXT_BUF_GROUP, priv->frm_grp);
        } else {
            auto fnClear = (MPP_RET(*)(MppBufferGroup))dlsym(g_mpp_handle,"mpp_buffer_group_clear");
            if(fnClear) fnClear(priv->frm_grp);
        }
        if(priv->frm_grp){
            auto fnLimit=(MPP_RET(*)(MppBufferGroup,size_t,RK_U32))dlsym(g_mpp_handle,"mpp_buffer_group_limit_config");
            if(fnLimit) fnLimit(priv->frm_grp, buf_size, 24);
        }
        priv->api->control(priv->ctx, MPP_DEC_SET_INFO_CHANGE_READY, nullptr);
        auto fnDeinit=(MPP_RET(*)(MppFrame*))dlsym(g_mpp_handle,"mpp_frame_deinit");
        if(fnDeinit) fnDeinit(&mpp_frame);
        return false; // 需下一包重试
    }
    // 正常帧
    RK_U32 w = ({ auto fn=(mpp_frame_get_width_fn)dlsym(g_mpp_handle,"mpp_frame_get_width"); fn?fn(mpp_frame):0; });
    RK_U32 h = ({ auto fn=(mpp_frame_get_height_fn)dlsym(g_mpp_handle,"mpp_frame_get_height"); fn?fn(mpp_frame):0; });
    RK_U32 hs = ({ auto fn=(mpp_frame_get_hor_stride_fn)dlsym(g_mpp_handle,"mpp_frame_get_hor_stride"); fn?fn(mpp_frame):0; });
    RK_U32 vs = ({ auto fn=(mpp_frame_get_ver_stride_fn)dlsym(g_mpp_handle,"mpp_frame_get_ver_stride"); fn?fn(mpp_frame):0; });
    // RK_U32 fmt = mpp_frame_get_fmt(mpp_frame);
    MppBuffer buf = ({ auto fn=(mpp_frame_get_buffer_fn)dlsym(g_mpp_handle,"mpp_frame_get_buffer"); fn?fn(mpp_frame):nullptr; });
    void* vir = nullptr;
    if(buf){
        auto fnPtr=(void*(*)(MppBuffer))dlsym(g_mpp_handle,"mpp_buffer_get_ptr");
        if(fnPtr) vir = fnPtr(buf);
    }
    if(!vir){ auto fnDeinit=(MPP_RET(*)(MppFrame*))dlsym(g_mpp_handle,"mpp_frame_deinit"); if(fnDeinit) fnDeinit(&mpp_frame); return false; }
    // 拷贝到 DecodedFrame（NV12： Y + UV）
    size_t y_size = hs * vs;
    size_t uv_size = hs * vs / 2;
    size_t total = y_size + uv_size;
    uint8_t* out = new uint8_t[total];
    memcpy(out, vir, total);
    frame.data = out; frame.data_uv = out + y_size;
    frame.width = w; frame.height = h; frame.pitch = hs; frame.pitch_uv = hs;
    frame.format = 0; frame.owns_data = true;
    auto fnDeinit=(MPP_RET(*)(MppFrame*))dlsym(g_mpp_handle,"mpp_frame_deinit");
    if(fnDeinit) fnDeinit(&mpp_frame);
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
