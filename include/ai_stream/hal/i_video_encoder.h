// include/ai_stream/hal/i_video_encoder.h
// 视频硬件编码抽象接口——隔离 MPP / NVENC / DVPP 等后端
//
// 与解码侧（IVideoCodec / VideoCodecFactory）对称：
// - 实现位于 src/hal/<platform>/，硬件代码不进入节点层
// - 节点（sink）只通过工厂按后端创建，输入统一 YUV420P 平面数据，
//   输出 AnnexB H.264（keyframe 标记 + pts/dts），由节点负责容器封装
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_stream {
namespace hal {

/**
 * @brief 编码配置
 */
struct VideoEncoderConfig {
    int width = 1920;
    int height = 1080;
    int bitrate_kbps = 4000;
    int fps = 25;
    int gop = 25;
    // 硬件后端忽略；FFmpeg 后端用作 avcodec_find_encoder_by_name 的名字
    // （libx264 / h264_nvenc / hevc_nvenc ...）
    std::string codec_name = "libx264";
    int device_id = 0;
};

/**
 * @brief 编码输出包（数据由实现持有，直到下次 encode/flush）
 */
struct EncodedPacket {
    const uint8_t* data = nullptr;
    size_t size = 0;
    int64_t pts = 0;
    int64_t dts = 0;
    bool keyframe = false;
};

/**
 * @brief 视频编码器抽象接口
 *
 * 输入：YUV420P 平面数据（连续 buffer，w*h*3/2 字节）
 * 输出：AnnexB H.264 码流
 */
class IVideoEncoder {
public:
    virtual ~IVideoEncoder() = default;

    /** 打开编码器；成功后 getExtradata() 返回 AVCC/序列头 */
    virtual bool open(const VideoEncoderConfig& config) = 0;

    /** 编码一帧 YUV420P；输出包追加到 packets */
    virtual bool encode(const uint8_t* yuv420p, size_t size, int64_t pts,
                        std::vector<EncodedPacket>& packets) = 0;

    /** 刷出内部缓冲 */
    virtual bool flush(std::vector<EncodedPacket>& packets) { (void)packets; return true; }

    virtual void close() = 0;

    /** 序列头（AVCC 格式 extradata，供容器封装；返回借用指针） */
    virtual const uint8_t* getExtradata(size_t& size) const = 0;

    virtual std::string getName() const = 0;
    virtual bool isAvailable() const = 0;
};

using VideoEncoderPtr = std::unique_ptr<IVideoEncoder>;

/**
 * @brief 视频编码器工厂
 */
class VideoEncoderFactory {
public:
    using Creator = std::function<VideoEncoderPtr()>;

    static VideoEncoderFactory& instance();

    void registerBackend(const std::string& name, Creator creator);
    /**
     * 按名称创建后端：
     * - "auto"      : mpp_h264 可用则优先，否则 ffmpeg_h264
     * - "mpp_h264"  : Rockchip MPP 硬编
     * - "ffmpeg_h264": FFmpeg 软编（codec_name 决定 libx264/nvenc 等）
     * 找不到/不可用返回 nullptr（节点回退默认路径）
     */
    VideoEncoderPtr create(const std::string& name);
    std::vector<std::string> getAvailableBackends() const;

private:
    VideoEncoderFactory() = default;
    std::unordered_map<std::string, Creator> creators_;
};

#define REGISTER_VIDEO_ENCODER(name, class_name) \
    static struct _VideoEncoderRegistrar_##class_name { \
        _VideoEncoderRegistrar_##class_name() { \
            VideoEncoderFactory::instance().registerBackend( \
                name, []() -> VideoEncoderPtr { \
                    return std::make_unique<class_name>(); \
                }); \
        } \
    } _video_encoder_registrar_##class_name;

} // namespace hal
} // namespace ai_stream
