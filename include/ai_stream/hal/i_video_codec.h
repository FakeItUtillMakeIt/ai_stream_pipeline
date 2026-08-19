// include/ai_stream/hal/i_video_codec.h
// 硬件编解码抽象接口——隔离 NVDEC / MPP / DVPP 等后端
#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <string>

namespace ai_stream {
namespace hal {

/**
 * @brief 解码后的帧数据
 */
struct DecodedFrame {
    uint8_t* data = nullptr;    // YUV 或 BGR 数据（主机内存）
    int width = 0;
    int height = 0;
    int pitch = 0;
    int format = 0;             // 像素格式（如 AV_PIX_FMT_NV12）
    bool owns_data = false;     // 是否需要释放

    // 多平面格式支持（如 NV12 的 Y 和 UV 平面）
    uint8_t* data_uv = nullptr; // UV 平面数据（NV12/NV21 等）
    int pitch_uv = 0;           // UV 平面 pitch
};

/**
 * @brief 硬件视频编解码抽象接口
 *
 * 封装硬件解码器（NVDEC / MPP / DVPP），
 * 各平台提供自己的实现。如果不需要硬件解码，
 * 可直接使用 FFmpeg 软件解码（ffmpeg_decode 节点）。
 */
class IVideoCodec {
public:
    virtual ~IVideoCodec() = default;

    /**
     * @brief 初始化解码器
     * @param codec_name 编码格式名称（如 "h264", "h265"）
     * @param extradata 编码器 extradata（SPS/PPS/VPS）
     * @param extradata_size extradata 大小
     * @return 是否初始化成功
     */
    virtual bool init(const std::string& codec_name,
                      const uint8_t* extradata = nullptr,
                      int extradata_size = 0) = 0;

    /**
     * @brief 解码一帧
     * @param packet_data 编码数据
     * @param packet_size 编码数据大小
     * @param frame 输出帧
     * @return 是否成功
     */
    virtual bool decode(const uint8_t* packet_data, int packet_size,
                        DecodedFrame& frame) = 0;

    /**
     * @brief 释放资源
     */
    virtual void release() = 0;

    /**
     * @brief 获取编解码器名称
     */
    virtual std::string getName() const = 0;

    /**
     * @brief 检查是否可用
     */
    virtual bool isAvailable() const = 0;
};

using VideoCodecPtr = std::unique_ptr<IVideoCodec>;

} // namespace hal
} // namespace ai_stream
