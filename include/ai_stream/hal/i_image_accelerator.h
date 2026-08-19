// include/ai_stream/hal/i_image_accelerator.h
// 图像加速抽象接口——隔离 NPP / CPU / RGA / DVPP 等后端
#pragma once

#include <cstdint>
#include <vector>
#include <string>
#include <memory>

namespace ai_stream {
namespace hal {

/**
 * @brief 边界框结构（用于绘制）
 */
struct BBox {
    float x, y, w, h;
    float confidence;
    int class_id;
    std::string class_name;
};

/**
 * @brief 图像加速抽象接口
 *
 * 封装 resize、normalize、NMS、OSD 绘制等图像操作，
 * 各平台提供自己的实现（NPP、CPU OpenCV、RGA、DVPP）。
 */
class IImageAccelerator {
public:
    virtual ~IImageAccelerator() = default;

    /**
     * @brief Resize + Normalize
     * @param src 源图像数据（RGB/BGR，uint8）
     * @param src_width 源宽度
     * @param src_height 源高度
     * @param dst 目标缓冲区（float32，NCHW 布局）
     * @param dst_width 目标宽度
     * @param dst_height 目标高度
     * @param mean 均值
     * @param std 标准差
     * @return 是否成功
     */
    virtual bool resizeNormalize(
        const uint8_t* src, int src_width, int src_height,
        float* dst, int dst_width, int dst_height,
        const std::vector<float>& mean,
        const std::vector<float>& std) = 0;

    /**
     * @brief 在 BGR 图像上绘制边界框
     * @param bgr BGR 图像数据
     * @param width 图像宽度
     * @param height 图像高度
     * @param pitch 行 pitch（字节）
     * @param boxes 要绘制的边界框
     * @return 是否成功
     */
    virtual bool drawBoxes(
        uint8_t* bgr, int width, int height, int pitch,
        const std::vector<BBox>& boxes) = 0;

    /**
     * @brief 非极大值抑制
     * @param boxes 输入/输出边界框（原地过滤）
     * @param iou_threshold IoU 阈值
     * @return 是否成功
     */
    virtual bool nms(
        std::vector<BBox>& boxes, float iou_threshold) = 0;

    /**
     * @brief 获取加速器名称（用于日志）
     */
    virtual std::string getName() const = 0;

    /**
     * @brief 检查加速器是否可用
     */
    virtual bool isAvailable() const = 0;
};

using ImageAcceleratorPtr = std::unique_ptr<IImageAccelerator>;

} // namespace hal
} // namespace ai_stream
