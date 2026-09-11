// src/nodes/sink/mp4_save.h
#pragma once

#include "ai_stream/core/queued_node.h"
#include "ai_stream/nodes/i_sink_node.h"
#include <memory>
#include <filesystem>

namespace ai_stream {
namespace nodes {

class EncoderBase;

class MP4SaveNode : public core::QueuedNode<ISinkNode> {
public:
    MP4SaveNode();
    ~MP4SaveNode() override;

    void setTarget(const std::string& target) override;
    void setEncodingParams(int bitrate, const std::string& encoder) override;
    void setOutputSize(int width, int height) override;
    bool isConnected() const override { return true; }

protected:
    bool onStartup() override;
    void onShutdown() override;
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;

private:
    bool initFileWriter();
    void closeFileWriter();

    std::string file_path_;
    int output_width_ = 0;
    int output_height_ = 0;
    int bitrate_ = 4000;
    std::string encoder_name_ = "libx264";

    std::unique_ptr<EncoderBase> encoder_;
    int64_t next_pts_ = 0;
};

} // namespace nodes
} // namespace ai_stream
