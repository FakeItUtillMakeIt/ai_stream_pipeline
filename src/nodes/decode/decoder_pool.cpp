// src/nodes/decode/decoder_pool.cpp
// 解码器上下文实现——封装 HAL VideoCodec 接口
#include "decoder_pool.h"
#include "3rd_party/log_mgr/log_mgr.h"

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
}

namespace ai_stream {
namespace nodes {

DecoderContext::DecoderContext() {
    bgr_frame = av_frame_alloc();
}

DecoderContext::~DecoderContext() {
    // HAL codec 通过 unique_ptr 自动释放
    codec.reset();

    if (d_bgr_buffer) {
#ifdef WITH_CUDA
        cudaFree(d_bgr_buffer);
#endif
        d_bgr_buffer = nullptr;
    }
    if (bgr_buffer) {
        av_free(bgr_buffer);
        bgr_buffer = nullptr;
    }
    if (bgr_frame) {
        av_frame_free(&bgr_frame);
    }
    if (sws_ctx) {
        sws_freeContext(sws_ctx);
        sws_ctx = nullptr;
    }
}

} // namespace nodes
} // namespace ai_stream
