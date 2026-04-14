// src/nodes/source/rtsp_source.h
#pragma once

#include "ai_stream/nodes/i_source_node.h"
#include <thread>
#include <atomic>
#include <string>

struct AVFormatContext;
struct AVPacket;

namespace ai_stream {
namespace nodes {

class RTSPSourceNode : public ISourceNode {
public:
    RTSPSourceNode();
    ~RTSPSourceNode() override;

    void setUrl(const std::string& url) override;
    std::string getUrl() const override;

    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void workerFunc();
    bool openInput();
    void closeInput();

    std::string url_;
    std::atomic<bool> running_{false};
    std::thread worker_;

    // FFmpeg 相关
    AVFormatContext* fmt_ctx_ = nullptr;
    int video_stream_index_ = -1;
    int codec_id_ = 0;                    // 编码格式 ID
    uint32_t my_stream_id_ = 0;
    
    // 统计信息
    int64_t total_frames_ = 0;
};

} // namespace nodes
} // namespace ai_stream