// src/nodes/decode/ffmpeg_decode.h
#pragma once

#include "ai_stream/nodes/i_decode_node.h"
#include "ai_stream/core/queued_node.h"
#include "decoder_pool.h"
#include <atomic>
#include <string>
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
    void setHwDecodeEnabled(bool enabled) override {use_hw_ = enabled;}
    bool isHwDecodeEnabled() override { return use_hw_.load(); }

private:
    std::shared_ptr<core::VideoFramePacket> decodePacket(
        std::shared_ptr<core::RawVideoPacket> raw_pkt,
        std::shared_ptr<DecoderContext> ctx);

    void saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num);

    DecoderPool pool_;
    std::string decoder_type_ = "h264";
    bool output_bgr_ = true;

    // 快照配置
    std::atomic<bool> snapshot_enabled_{false};
    std::atomic<int> snapshot_interval_{100};
    std::string snapshot_dir_ = "./snapshots";

    int snapshot_count_ = 0;
    int frame_count_ = 0;
    std::atomic<bool> use_hw_{false};
#ifdef WITH_CUDA
    cudaStream_t cuda_stream_ = nullptr;  // 用于 NV12→BGR 转换
#endif
};

} // namespace nodes
} // namespace ai_stream