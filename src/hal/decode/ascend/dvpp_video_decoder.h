// src/hal/decode/ascend/dvpp_video_decoder.h
// 华为 Ascend DVPP VDEC 硬件解码（acl vdec API）
//
// 输入：AnnexB H.264/H.265 码流（RTSP/MP4 的 AVCC 会先转 AnnexB）
// 输出：主机侧紧致 NV12，供节点层 NV12→BGR 或 NV12 直通消费。
//
// DVPP 的回调由 report 线程 + aclrtProcessReport() 驱动，因此每个解码器
// 持有一个独立 context / report 线程；decode() 等待回调完成一帧。
#pragma once

#include "ai_stream/hal/i_video_decoder.h"

#include <acl/acl.h>
#include <acl/ops/acl_dvpp.h>

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <pthread.h>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

class DvppVideoDecoder : public IVideoDecoder {
public:
    DvppVideoDecoder() = default;
    ~DvppVideoDecoder() override;

    bool init(const std::string& codec_name,
              const uint8_t* extradata = nullptr,
              int extradata_size = 0) override;

    bool decode(const uint8_t* packet_data, int packet_size,
                DecodedFrame& frame) override;

    void release() override;

    std::string getName() const override { return "DVPP (Ascend)"; }
    bool isAvailable() const override;

    void setSourceResolution(int width, int height) override {
        source_width_ = width;
        source_height_ = height;
    }

private:
    bool initDvpp();
    void cleanup();
    void reportLoop();
    static void* reportThreadEntry(void* arg);
    static void vdecCallback(acldvppStreamDesc* input, acldvppPicDesc* output,
                             void* userData);

    // AVCC(AVCDecoderConfigurationRecord) → SPS/PPS AnnexB；兼容已经是 AnnexB 的 extradata
    bool parseExtradata();
    // 单包 AVCC → AnnexB；如果输入本身是 AnnexB 则原样透传
    bool toAnnexb(const uint8_t* data, int size, std::vector<uint8_t>& out) const;
    // 判断 AnnexB 包首个 NAL 是否为 IDR（H.264 type 5 / H.265 type 19/20）
    bool isIdr(const std::vector<uint8_t>& annexb) const;

    std::string codec_name_;
    std::vector<uint8_t> extradata_;
    std::vector<uint8_t> sps_pps_annexb_;
    int nal_length_size_ = 4;
    acldvppStreamFormat en_type_ = H264_HIGH_LEVEL;

    int source_width_ = 0;
    int source_height_ = 0;
    int out_width_ = 0;
    int out_height_ = 0;
    size_t output_buffer_size_ = 0;
    uint32_t channel_id_ = 10;

    aclvdecChannelDesc* channel_ = nullptr;
    aclrtContext context_ = nullptr;
    aclrtStream stream_ = nullptr;
    pthread_t report_thread_{};
    std::atomic<bool> report_running_{false};
    bool initialized_ = false;
    bool sps_pps_sent_ = false;

    std::mutex frame_mutex_;
    std::condition_variable frame_cv_;
    std::deque<std::vector<uint8_t>> frames_;
    int frame_width_ = 0;
    int frame_height_ = 0;
    std::vector<uint8_t> frame_buf_;

    uint64_t sent_packets_ = 0;
    uint64_t recv_frames_ = 0;
    uint64_t timeout_count_ = 0;
};

} // namespace hal
} // namespace ai_stream
