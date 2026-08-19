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
 * @brief Resize+Normalize 参数
 */
struct ResizeNormalizeParams {
    int src_width = 0;
    int src_height = 0;
    size_t src_pitch = 0;       // GPU 输入时的行 pitch（字节）
    int dst_width = 0;
    int dst_height = 0;
    bool keep_aspect_ratio = false;  // 是否保持宽高比（letterbox）
    std::vector<float> mean{0.0f, 0.0f, 0.0f};
    std::vector<float> std{1.0f, 1.0f, 1.0f};
    void* stream = nullptr;     // CUDA stream（nullptr 表示同步）
};

/**
 * @brief Letterbox 输出参数
 */
struct LetterboxResult {
    int letter_w = 0;
    int letter_h = 0;
    int pad_x = 0;
    int pad_y = 0;
    float scale = 1.0f;
};

/**
 * @brief 绘制参数
 */
struct DrawParams {
    uint8_t* bgr = nullptr;     // BGR 图像数据
    int width = 0;
    int height = 0;
    int pitch = 0;              // 行 pitch（字节）
    bool is_gpu = false;        // 数据是否在 GPU 上
    void* stream = nullptr;     // CUDA stream
    // 绘制选项
    int box_color_b = 0;
    int box_color_g = 255;
    int box_color_r = 0;
    int font_thickness = 1;
    bool show_confidence = true;
    std::vector<int> class_filter;
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
     * @param params 参数
     * @param dst 目标缓冲区（float32，NCHW 布局，GPU 内存）
     * @param letter 输出 letterbox 参数（如果 keep_aspect_ratio=true）
     * @return 是否成功
     */
    virtual bool resizeNormalize(
        const uint8_t* src,
        const ResizeNormalizeParams& params,
        float* dst,
        LetterboxResult* letter = nullptr) = 0;

    /**
     * @brief 在 BGR 图像上绘制边界框
     * @param boxes 要绘制的边界框
     * @param draw 绘制参数
     * @return 是否成功
     */
    virtual bool drawBoxes(
        const std::vector<BBox>& boxes,
        const DrawParams& draw) = 0;

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
