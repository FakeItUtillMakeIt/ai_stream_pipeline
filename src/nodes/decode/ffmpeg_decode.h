// src/nodes/decode/ffmpeg_decode.h
#pragma once

#include "ai_stream/nodes/i_decode_node.h"
#include "decoder_pool.h"
#include <atomic>
#include <string>

namespace ai_stream {
namespace nodes {

class FFmpegDecodeNode : public IDecodeNode {
public:
    FFmpegDecodeNode();
    ~FFmpegDecodeNode() override;

    // IDecodeNode 接口实现
    void setDecoderType(const std::string& codec_name) override;
    void setOutputBGR(bool enable) override;
    size_t getActiveDecoderCount() const override;

    // Node 接口
    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

    // 快照配置
    void setSnapshotEnabled(bool enabled);
    void setSnapshotInterval(int interval);
    void setSnapshotDir(const std::string& dir);

private:
    std::shared_ptr<core::VideoFramePacket> decodePacket(
        std::shared_ptr<core::RawVideoPacket> raw_pkt,
        std::shared_ptr<DecoderContext> ctx);
    
    void saveSnapshot(std::shared_ptr<core::VideoFramePacket> frame, int frame_num);

    DecoderPool pool_;
    std::string decoder_type_ = "h264";
    bool output_bgr_ = true;
    std::atomic<bool> running_{false};
    
    // 快照配置
    std::atomic<bool> snapshot_enabled_{false};
    std::atomic<int> snapshot_interval_{100};
    std::string snapshot_dir_ = "./snapshots";
    
    int snapshot_count_ = 0;
    int frame_count_ = 0;
};

} // namespace nodes
} // namespace ai_stream