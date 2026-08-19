// src/nodes/decode/ffmpeg_decode.h
// 解码节点——使用 VideoCodecFactory，支持多后端（NVDEC/MPP/DVPP/FFmpeg）
#pragma once

#include "ai_stream/nodes/i_decode_node.h"
#include "ai_stream/core/queued_node.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "decoder_pool.h"
#include <atomic>
#include <string>
#include <unordered_map>
#include <mutex>
#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace ai_stream {
namespace nodes {

class FFmpegDecodeNode : public core::QueuedNode<IDecodeNode> {
public:
    FFmpegDecodeNode();
    ~FFmpegDecodeNode() override;

    // IDecodeNode 接口实现
    void setDecoderType(const std::string& codec_name) override;
    void setOutputBGR(bool enable) override;
    size_t getActiveDecoderCount() const override;

    // QueuedNode 接口
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;
    bool onStartup() override;
    void onShutdown() override;

    // 快照配置
    void setSnapshotEnabled(bool enabled) override;
    void setSnapshotInterval(int interval) override;
    void setSnapshotDir(const std::string& dir) override;
    void setHwDecodeEnabled(bool enabled) override;
    bool isHwDecodeEnabled() override;

    // 设置视频编解码后端
    void setVideoCodecBackend(hal::VideoCodecBackend backend);

private:
    std::shared_ptr<DecoderContext> getOrCreateDecoder(
        uint32_t stream_id, int codec_id,
        const uint8_t* extradata, int extradata_size);

    std::shared_ptr<core::VideoFramePacket> decodePacket(
        std::shared_ptr<core::RawVideoPacket> raw_pkt,
        std::shared_ptr<DecoderContext> ctx);

    void saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num);

    // 解码器管理（每个 stream_id 一个）
    std::unordered_map<uint32_t, std::shared_ptr<DecoderContext>> decoders_;
    mutable std::mutex mutex_;
    std::atomic<size_t> active_decoders_{0};

    // 配置
    hal::VideoCodecBackend codec_backend_ = hal::VideoCodecBackend::AUTO;
    std::string decoder_type_ = "h264";
    bool output_bgr_ = true;

    // 快照配置
    std::atomic<bool> snapshot_enabled_{false};
    std::atomic<int> snapshot_interval_{100};
    std::string snapshot_dir_ = "./snapshots";

    int snapshot_count_ = 0;
    int frame_count_ = 0;
    std::atomic<bool> use_hw_{false};
    int64_t in_time_ms_ = 0;
#ifdef WITH_CUDA
    cudaStream_t cuda_stream_ = nullptr;  // 用于 NV12→BGR 转换
#endif
};

} // namespace nodes
} // namespace ai_stream
