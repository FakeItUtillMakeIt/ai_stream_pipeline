// src/hal/encode/ascend/dvpp_video_encoder.h
// 华为 Ascend DVPP VENC 硬件编码（H.264）
//
// 输入：连续 YUV420P；内部转 NV12（16 字节对齐）上传到设备。
// 输出：AnnexB H.264 包 + AnnexB extradata（SPS/PPS）。
//
// VENC 是回调式的：每个编码器持有 context/report 线程，encode() 发送一帧后
// 等待回调。open() 会先编码一帧灰色图来提取 SPS/PPS（供 muxer 写头），
// 该帧丢弃；首个真实帧强制 IDR。
#pragma once

#include "ai_stream/hal/i_video_encoder.h"

#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <pthread.h>
#include <vector>

namespace ai_stream {
namespace hal {

class DvppVideoEncoder : public IVideoEncoder {
public:
    DvppVideoEncoder() = default;
    ~DvppVideoEncoder() override;

    bool open(const VideoEncoderConfig& config) override;
    bool encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                std::vector<EncodedPacket>& packets) override;
    bool flush(std::vector<EncodedPacket>& packets) override;
    void close() override;

    const uint8_t* getExtradata(size_t& size) const override {
        size = extradata_.size();
        return extradata_.empty() ? nullptr : extradata_.data();
    }

    std::string getName() const override { return "DVPP VENC H.264 (Ascend)"; }
    bool isAvailable() const override;

private:
    struct OutPacket {
        std::vector<uint8_t> data;
        int64_t pts = 0;
        bool keyframe = false;
    };

    bool initVenc();
    bool primeExtradata();
    void cleanup();
    void reportLoop();
    static void* reportThreadEntry(void* arg);
    static void vencCallback(acldvppPicDesc* input, acldvppStreamDesc* output,
                             void* userData);

    void fillNv12(const uint8_t* yuv420p, std::vector<uint8_t>& nv12) const;
    static bool extractSpsPps(const uint8_t* data, size_t size,
                              std::vector<uint8_t>& out);
    static bool isIdr(const uint8_t* data, size_t size);

    int width_ = 0;
    int height_ = 0;
    int aligned_w_ = 0;
    int aligned_h_ = 0;
    int fps_ = 25;
    int gop_ = 25;
    int bitrate_kbps_ = 4000;
    size_t frame_size_ = 0;

    aclvencChannelDesc* channel_ = nullptr;
    aclvencFrameConfig* frame_config_ = nullptr;
    void* input_dev_ = nullptr;
    acldvppPicDesc* input_pic_ = nullptr;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    pthread_t report_thread_{};
    std::atomic<bool> report_running_{false};
    bool opened_ = false;
    bool first_frame_ = true;
    bool priming_ = false;
    bool flushed_ = false;

    std::mutex packet_mutex_;
    std::condition_variable packet_cv_;
    std::deque<OutPacket> packets_;
    std::atomic<int64_t> current_pts_{0};

    std::vector<uint8_t> extradata_;
    std::vector<std::vector<uint8_t>> staging_;
};

} // namespace hal
} // namespace ai_stream
