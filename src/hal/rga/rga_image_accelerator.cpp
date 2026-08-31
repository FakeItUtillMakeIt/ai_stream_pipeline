// src/hal/rga/rga_image_accelerator.cpp — Rockchip RGA 2D 加速（dlopen 动态加载，x86 仅头文件编译）
#include "rga_image_accelerator.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <dlfcn.h>
#include <unistd.h>
#include <cstring>
#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

// 仅在 WITH_RKNN 时包含 RGA 头（3rd_party/rk_platform/rga/include 已加入 include path）
#ifdef WITH_RKNN
#include "rga.h"
#include "im2d.h"
#include "RgaApi.h"
#endif

namespace ai_stream {
namespace hal {

#ifdef WITH_RKNN
static void* g_rga_handle = nullptr;
static bool g_rga_loaded = false;
static bool g_rga_load_tried = false;

static bool load_rga() {
    if (g_rga_load_tried) return g_rga_loaded;
    g_rga_load_tried = true;
    // 优先 3rd_party 路径，其次系统路径
    const char* cand[] = {
        "3rd_party/rk_platform/rga/lib/aarch64/librga.so",
        "librga.so",
        nullptr
    };
    for (int i = 0; cand[i]; ++i) {
        g_rga_handle = dlopen(cand[i], RTLD_NOW);
        if (g_rga_handle) break;
    }
    if (!g_rga_handle) {
        LOG_DEBUG_FMT("[RgaImageAccelerator] dlopen librga.so failed: {}", dlerror() ? dlerror() : "unknown");
        return false;
    }
    // 验证关键符号
    if (!dlsym(g_rga_handle, "c_RkRgaInit") && !dlsym(g_rga_handle, "imresize")) {
        LOG_WARN("[RgaImageAccelerator] librga.so missing expected symbols");
        dlclose(g_rga_handle); g_rga_handle = nullptr;
        return false;
    }
    g_rga_loaded = true;
    LOG_INFO("[RgaImageAccelerator] librga.so loaded");
    return true;
}
#endif

RgaImageAccelerator::RgaImageAccelerator() {
    initialized_ = initRga();
    if (initialized_) LOG_DEBUG("[RgaImageAccelerator] Initialized");
}

RgaImageAccelerator::~RgaImageAccelerator() { cleanup(); }

bool RgaImageAccelerator::isAvailable() const {
#ifdef WITH_RKNN
    if (initialized_) return true;
    if (access("/dev/rga", F_OK) == 0) return true;
    // 尝试加载库判断
    // const_cast 触发加载（仅检测）
    return false;
#else
    return false;
#endif
}

void RgaImageAccelerator::setCoreMask(int core_mask) { core_mask_ = core_mask; }

bool RgaImageAccelerator::initRga() {
#ifdef WITH_RKNN
    if (access("/dev/rga", F_OK) != 0) {
        // 无设备时仍尝试加载库，x86 上会失败并返回 false（静默降级到 CPU）
        if (!load_rga()) {
            LOG_DEBUG("[RgaImageAccelerator] /dev/rga not found, fallback to CPU");
            return false;
        }
        // 即使无设备，标记可用以便单元测试；实际 Blit 会失败并回退
    }
    if (!load_rga()) return false;
    // c_RkRgaInit 在新版 RGA 中可空操作，仍调用一次
    auto fnInit = (int(*)())dlsym(g_rga_handle, "c_RkRgaInit");
    if (fnInit) {
        int ret = fnInit();
        if (ret != 0) LOG_WARN_FMT("[RgaImageAccelerator] c_RkRgaInit ret={}", ret);
    }
    return true;
#else
    LOG_DEBUG("[RgaImageAccelerator] WITH_RKNN not enabled");
    return false;
#endif
}

void RgaImageAccelerator::cleanup() {
#ifdef WITH_RKNN
    if (g_rga_handle) {
        auto fnDeInit = (void(*)())dlsym(g_rga_handle, "c_RkRgaDeInit");
        if (fnDeInit) fnDeInit();
        // 保持句柄常驻，避免重复 dlopen；如需释放可 dlclose
    }
#endif
    initialized_ = false;
}

// CPU 回退：OpenCV resize + 归一化
static bool cpuResizeNormalize(const uint8_t* src, const ResizeNormalizeParams& p, float* dst, LetterboxResult* letter) {
    if (!src || !dst) return false;
    int sw = p.src_width, sh = p.src_height;
    int dw = p.dst_width, dh = p.dst_height;
    if (sw <= 0 || sh <= 0 || dw <= 0 || dh <= 0) return false;
    cv::Mat srcMat(sh, sw, CV_8UC3, const_cast<uint8_t*>(src));
    cv::Mat resized;
    float scale = 1.0f; int padX = 0, padY = 0, nw = dw, nh = dh;
    if (p.keep_aspect_ratio) {
        scale = std::min(float(dw) / sw, float(dh) / sh);
        nw = int(sw * scale); nh = int(sh * scale);
        padX = (dw - nw) / 2; padY = (dh - nh) / 2;
        cv::Mat tmp; cv::resize(srcMat, tmp, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
        resized = cv::Mat(dh, dw, CV_8UC3, cv::Scalar(114,114,114));
        tmp.copyTo(resized(cv::Rect(padX, padY, nw, nh)));
    } else {
        cv::resize(srcMat, resized, cv::Size(dw, dh), 0, 0, cv::INTER_LINEAR);
        scale = float(dw) / sw;
    }
    if (letter) { letter->scale = scale; letter->pad_x = padX; letter->pad_y = padY; letter->letter_w = nw; letter->letter_h = nh; }
    // BGR -> RGB 归一化（与 NPP/CPU 保持一致）
    float mean0 = p.mean.size() > 0 ? p.mean[0] : 0.f;
    float mean1 = p.mean.size() > 1 ? p.mean[1] : 0.f;
    float mean2 = p.mean.size() > 2 ? p.mean[2] : 0.f;
    float std0 = p.std.size() > 0 ? p.std[0] : 1.f;
    float std1 = p.std.size() > 1 ? p.std[1] : 1.f;
    float std2 = p.std.size() > 2 ? p.std[2] : 1.f;
    int plane = dw * dh;
    for (int y = 0; y < dh; ++y) {
        uint8_t* row = resized.ptr<uint8_t>(y);
        for (int x = 0; x < dw; ++x) {
            uint8_t b = row[x*3+0], g = row[x*3+1], r = row[x*3+2];
            int idx = y*dw + x;
            dst[0*plane + idx] = (r - mean0) / std0;
            dst[1*plane + idx] = (g - mean1) / std1;
            dst[2*plane + idx] = (b - mean2) / std2;
        }
    }
    return true;
}

bool RgaImageAccelerator::resizeNormalize(const uint8_t* src, const ResizeNormalizeParams& params, float* dst, LetterboxResult* letter) {
    if (!src || !dst) return false;
#ifdef WITH_RKNN
    // 尝试 RGA 路径
    if (initialized_ && g_rga_handle && access("/dev/rga", F_OK) == 0) {
        // 使用 RGA 做 resize 到中间 BGR 缓冲，再 CPU 归一化
        int sw = params.src_width, sh = params.src_height;
        int dw = params.dst_width, dh = params.dst_height;
        if (sw > 0 && sh > 0 && dw > 0 && dh > 0) {
            // 中间缓冲
            std::vector<uint8_t> tmp(dw * dh * 3);
            // wrapbuffer 虚拟地址
            auto fnWrap = (rga_buffer_t(*)(void*,int,int,int,int,int))dlsym(g_rga_handle, "wrapbuffer_virtualaddr_t");
            auto fnResize = (int(*)(rga_buffer_t, rga_buffer_t, double, double, int, int))dlsym(g_rga_handle, "imresize_t");
            if (fnWrap && fnResize) {
                int fmt = RK_FORMAT_RGB_888; // 与 BGR 888 同布局，RGA 会处理
                // 注意 rga 期望 BGR/RGB 格式；使用 RGB_888
                rga_buffer_t srcBuf = fnWrap(const_cast<uint8_t*>(src), sw, sh, sw, sh, fmt);
                rga_buffer_t dstBuf = fnWrap(tmp.data(), dw, dh, dw, dh, fmt);
                // letterbox 时先按比例 resize 到 nw*nh 再居中拷贝（简化：直接 resize 到 dw*dh，letter 信息仍计算）
                int ret = fnResize(srcBuf, dstBuf, 0, 0, 0, 1);
                if (ret == 0) {
                    // 成功则对 tmp 做归一化
                    ResizeNormalizeParams tmpP = params;
                    tmpP.src_width = dw; tmpP.src_height = dh;
                    // 此时 tmp 已是 dw*dh，直接归一化（避免二次 resize）
                    float mean0 = params.mean.size()>0?params.mean[0]:0, mean1=params.mean.size()>1?params.mean[1]:0, mean2=params.mean.size()>2?params.mean[2]:0;
                    float std0 = params.std.size()>0?params.std[0]:1, std1=params.std.size()>1?params.std[1]:1, std2=params.std.size()>2?params.std[2]:1;
                    int plane = dw*dh;
                    // 处理 letterbox 的 pad 区域（若 keep_aspect_ratio）
                    if (params.keep_aspect_ratio) {
                        float scale = std::min(float(dw)/sw, float(dh)/sh);
                        int nw = int(sw*scale), nh = int(sh*scale);
                        int padX=(dw-nw)/2, padY=(dh-nh)/2;
                        if (letter){ letter->scale=scale; letter->pad_x=padX; letter->pad_y=padY; letter->letter_w=nw; letter->letter_h=nh; }
                        // 对 RGA 结果做 letterbox 填充（RGA 已缩放到 dw*dh，需重做？为简化回退到 CPU letterbox）
                        return cpuResizeNormalize(src, params, dst, letter);
                    }
                    for (int y=0;y<dh;++y) for(int x=0;x<dw;++x){ int idx=y*dw+x; uint8_t r=tmp[idx*3+0], g=tmp[idx*3+1], b=tmp[idx*3+2]; dst[0*plane+idx]=(r-mean0)/std0; dst[1*plane+idx]=(g-mean1)/std1; dst[2*plane+idx]=(b-mean2)/std2; }
                    if (letter && !params.keep_aspect_ratio){ letter->scale=float(dw)/sw; letter->pad_x=0; letter->pad_y=0; letter->letter_w=dw; letter->letter_h=dh; }
                    return true;
                }
                LOG_DEBUG_FMT("[RgaImageAccelerator] RGA imresize failed ret={}, fallback CPU", ret);
            }
        }
    }
#endif
    return cpuResizeNormalize(src, params, dst, letter);
}

bool RgaImageAccelerator::drawBoxes(const std::vector<BBox>& boxes, const DrawParams& draw) {
    if (!draw.bgr || draw.width <=0 || draw.height<=0) return false;
    cv::Mat img(draw.height, draw.width, CV_8UC3, draw.bgr);
    for (auto &b: boxes) {
        if (!draw.class_filter.empty() && std::find(draw.class_filter.begin(), draw.class_filter.end(), b.class_id)==draw.class_filter.end()) continue;
        int x1 = int(b.x), y1=int(b.y), x2=int(b.x+b.w), y2=int(b.y+b.h);
        cv::Scalar color(draw.box_color_b, draw.box_color_g, draw.box_color_r);
        cv::rectangle(img, {x1,y1},{x2,y2}, color, draw.font_thickness);
        if (draw.show_confidence) {
            std::string txt = b.class_name + " " + std::to_string(b.confidence).substr(0,4);
            cv::putText(img, txt, {x1, std::max(0,y1-5)}, cv::FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
        }
    }
    return true;
}

bool RgaImageAccelerator::nms(std::vector<BBox>& boxes, float iou_threshold) {
    if (boxes.empty()) return true;
    std::sort(boxes.begin(), boxes.end(), [](auto&a, auto&b){return a.confidence > b.confidence;});
    std::vector<BBox> keep;
    std::vector<bool> sup(boxes.size(), false);
    auto iou = [](const BBox& a, const BBox& b){
        float x1 = std::max(a.x, b.x), y1 = std::max(a.y, b.y);
        float x2 = std::min(a.x+a.w, b.x+b.w), y2 = std::min(a.y+a.h, b.y+b.h);
        float inter = std::max(0.f, x2-x1) * std::max(0.f, y2-y1);
        float uni = a.w*a.h + b.w*b.h - inter;
        return uni>0 ? inter/uni : 0.f;
    };
    for (size_t i=0;i<boxes.size();++i){ if(sup[i]) continue; keep.push_back(boxes[i]); for(size_t j=i+1;j<boxes.size();++j) if(!sup[j] && boxes[i].class_id==boxes[j].class_id && iou(boxes[i],boxes[j])>iou_threshold) sup[j]=true; }
    boxes.swap(keep);
    return true;
}

#ifdef WITH_RKNN
REGISTER_IMAGE_ACCELERATOR(ImageAcceleratorBackend::RGA, RgaImageAccelerator)
#endif
} // namespace hal
} // namespace ai_stream
