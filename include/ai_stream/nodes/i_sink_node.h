// include/ai_stream/nodes/i_sink_node.h
#pragma once

#include "ai_stream/core/node.h"
#include <string>

namespace ai_stream {
namespace nodes {

/**
 * @brief 输出节点接口（Sink）
 * 
 * 负责将处理后的视频流输出到目标，例如 RTMP 推流、本地文件保存或显示窗口。
 */
class ISinkNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 设置输出目标
     * @param target 输出目标地址：
     *               - RTMP 地址: "rtmp://server/live/stream"
     *               - 本地文件: "file:///path/to/output.mp4"
     *               - 窗口显示: "display://0"
     */
    virtual void setTarget(const std::string& target) = 0;

    /**
     * @brief 设置编码参数（如比特率、编码器）
     * @param bitrate 视频比特率 (kbps)
     * @param encoder 编码器名称 ("libx264", "h264_nvenc" 等)
     */
    virtual void setEncodingParams(int bitrate, const std::string& encoder) = 0;

    /**
     * @brief 设置输出分辨率（若不设置则保持原始尺寸）
     */
    virtual void setOutputSize(int width, int height) = 0;

    /**
     * @brief 获取当前输出是否连接正常（仅对网络流有效）
     */
    virtual bool isConnected() const = 0;
};

} // namespace nodes
} // namespace ai_stream