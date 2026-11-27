// src/nodes/common/cpu_frame.h
// NV12 直通解码后，部分下游（证据/录像/推流）仍需要 CPU BGR。
// 这里统一提供"按需把 d_bgr_ptr 下载成 cv::Mat"的兜底，避免各节点重复实现。
#pragma once

#include "ai_stream/core/packet.h"

#include <opencv2/core.hpp>
#include <memory>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace ai_stream {
namespace nodes {

/**
 * @brief 确保帧带有可用的 CPU BGR cv::Mat
 *
 * 正常情况下 decode 已经产出 mat；NV12 直通模式下 mat 为空，但 preprocess
 * 会暴露全分辨率 GPU BGR（d_bgr_ptr）。仅在确实需要 CPU 图像时才下载，
 * 避免把 GPU 链路重新拉回 CPU。
 *
 * @return true 表示 frame->mat 可用
 */
inline bool ensureCpuBgr(const std::shared_ptr<core::VideoFramePacket>& frame) {
    if (!frame) return false;
    if (frame->mat && !frame->mat->empty()) return true;

#ifdef WITH_CUDA
    if (frame->d_bgr_ptr && frame->d_bgr_width > 0 && frame->d_bgr_height > 0) {
        const int w = frame->d_bgr_width;
        const int h = frame->d_bgr_height;
        const size_t pitch = frame->d_bgr_pitch > 0
                                 ? static_cast<size_t>(frame->d_bgr_pitch)
                                 : static_cast<size_t>(w) * 3;
        cv::Mat m(h, w, CV_8UC3);
        if (cudaMemcpy2D(m.data, m.step, frame->d_bgr_ptr, pitch,
                         static_cast<size_t>(w) * 3, h,
                         cudaMemcpyDeviceToHost) != cudaSuccess) {
            return false;
        }
        frame->mat = std::make_shared<cv::Mat>(std::move(m));
        if (!frame->source_mat) frame->source_mat = frame->mat;
        frame->width = w;
        frame->height = h;
        frame->channels = 3;
        return true;
    }
#endif
    return false;
}

} // namespace nodes
} // namespace ai_stream
