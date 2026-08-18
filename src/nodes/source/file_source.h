// src/nodes/source/file_source.h
#pragma once

#include "ai_stream/nodes/i_source_node.h"
#include <thread>
#include <atomic>
#include <string>

struct AVFormatContext;
struct AVPacket;

namespace ai_stream {
namespace nodes {

/**
 * @brief 本地文件源节点
 *
 * 从本地视频文件（mp4/mkv/flv 等）读取编码帧并广播 RawVideoPacket，
 * 用于无 RTSP 流时的调试与性能基准测试。
 *
 * 额外 JSON 参数（在 ISourceNode 的 url/skip_frames 之外）：
 *   "loop":     bool，播完后是否循环（默认 true）
 *   "realtime": bool，是否按原始帧率节奏推送（默认 true）
 */
class FileSourceNode : public ISourceNode {
public:
    FileSourceNode();
    ~FileSourceNode() override;

    void setUrl(const std::string& url) override;
    void setSourceId(const std::string& id) override;
    void setSkipFrames(int skip_frames) override;
    std::string getUrl() const override;

    bool configure(const std::string& node_id, const nlohmann::json& params) override;

    void setLoopEnabled(bool enabled) { loop_enabled_ = enabled; }
    void setRealtimeEnabled(bool enabled) { realtime_ = enabled; }

    bool start() override;
    void stop() override;
    bool isRunning() const override { return running_.load(); }
    void pushData(std::shared_ptr<core::BasePacket> packet) override;

private:
    void workerFunc();
    bool openInput();
    void closeInput();

    std::string url_;
    std::string source_id_;
    std::atomic<bool> running_{false};
    std::atomic<bool> loop_enabled_{true};
    std::atomic<bool> realtime_{true};
    std::thread worker_;

    AVFormatContext* fmt_ctx_ = nullptr;
    int video_stream_index_ = -1;
    int codec_id_ = 0;
    std::vector<uint8_t> extradata_;
    uint32_t my_stream_id_ = 0;

    std::atomic<int64_t> total_frames_{0};
    uint8_t skip_frames_ = 1;
};

} // namespace nodes
} // namespace ai_stream
