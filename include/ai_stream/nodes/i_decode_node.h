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

    /**
     * @brief 是否启用快照
     * @param packet 包含原始视频数据的数据包
     */
    virtual void setSnapshotEnabled(bool enabled) = 0;

    /**
     * @brief 设置快照间隔
     */
    virtual void setSnapshotInterval(int interval) = 0;

    /**
     * @brief 设置快照保存目录
     *  @param packet 待处理的数据包
     */
    virtual void setSnapshotDir(const std::string& dir) = 0;

    virtual void setHwDecodeEnabled(bool enabled) = 0;
    virtual bool isHwDecodeEnabled() = 0;

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        (void)node_id;
        if (params.contains("codec")) {
            setDecoderType(params["codec"].get<std::string>());
        }
        if (params.contains("output_bgr")) {
            setOutputBGR(params["output_bgr"].get<bool>());
        }
        if (params.contains("hw_decoder")) {
            setHwDecodeEnabled(params["hw_decoder"].get<bool>());
        }
        if (params.contains("snapshot")) {
            const auto& snapshot_cfg = params["snapshot"];
            if (snapshot_cfg.contains("enabled")) {
                setSnapshotEnabled(snapshot_cfg["enabled"].get<bool>());
            }
            if (snapshot_cfg.contains("interval")) {
                setSnapshotInterval(snapshot_cfg["interval"].get<int>());
            }
            if (snapshot_cfg.contains("dir")) {
                setSnapshotDir(snapshot_cfg["dir"].get<std::string>());
            }
        }
        return true;
    }
};

} // namespace nodes
} // namespace ai_stream