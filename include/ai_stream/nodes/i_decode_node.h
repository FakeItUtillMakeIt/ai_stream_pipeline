// include/ai_stream/nodes/i_decode_node.h
#pragma once

#include "ai_stream/core/node.h"
#include <string>

namespace ai_stream {
namespace nodes {

/**
 * @brief 解码节点接口
 * 
 * 负责将原始视频数据（如 H.264/H.265 字节流）解码为可处理的图像帧。
 * 支持多路流并行解码，通过 stream_id 区分不同源。
 */
class IDecodeNode : public core::Node {
public:
    using core::Node::Node;

    /**
     * @brief 设置解码器类型（如 "h264_cuvid"、"h264"）
     * @param codec_name 解码器名称
     */
    virtual void setDecoderType(const std::string& codec_name) = 0;

    /**
     * @brief 设置是否输出 BGR 格式（默认可能为 YUV，方便 OpenCV 处理）
     * @param enable true 表示输出 BGR，false 表示保持原始格式
     */
    virtual void setOutputBGR(bool enable) = 0;

    /**
     * @brief 获取当前解码器池的大小（用于监控多流并行状态）
     * @return 当前活跃的解码器实例数量
     */
    virtual size_t getActiveDecoderCount() const = 0;
};

} // namespace nodes
} // namespace ai_stream