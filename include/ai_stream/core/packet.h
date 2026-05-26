// include/ai_stream/core/packet.h
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <opencv2/core/mat.hpp>

namespace ai_stream {
namespace core {

// ========== COCO骨架定义 ==========
static const std::vector<std::pair<int, int>> SKELETON = {
    {0, 1}, {0, 2}, {1, 3}, {2, 4},           // 脸部
    {5, 6}, {5, 7}, {7, 9}, {6, 8}, {8, 10}, // 上肢
    {5, 11}, {6, 12}, {11, 12},              // 躯干
    {11, 13}, {13, 15}, {12, 14}, {14, 16}  // 下肢
};

static const cv::Scalar COLOR_HEAD(0, 255, 0);      // 绿
static const cv::Scalar COLOR_ARMS(255, 0, 0);      // 蓝
static const cv::Scalar COLOR_BODY(0, 165, 255);    // 橙
static const cv::Scalar COLOR_LEGS(0, 0, 255);      // 红

static const cv::Scalar KPT_COLORS[17] = {
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),
    cv::Scalar(0, 255, 0), cv::Scalar(0, 255, 0),     // 0-4 脸部绿
    cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0),     // 5-6 肩蓝
    cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0),     // 7-8 肘蓝
    cv::Scalar(255, 0, 0), cv::Scalar(255, 0, 0),     // 9-10 手腕蓝
    cv::Scalar(0, 165, 255), cv::Scalar(0, 165, 255), // 11-12 髋橙
    cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255),     // 13-14 膝红
    cv::Scalar(0, 0, 255), cv::Scalar(0, 0, 255)      // 15-16 踝红
};


/**
 * @brief 数据包类型枚举
 */
enum class PacketType : uint8_t {
    UNKNOWN = 0,
    RAW_VIDEO,      // 原始编码视频数据 (H264/H265 字节流)
    DECODED_FRAME,  // 解码后的图像帧 (YUV/RGB)
    DETECTION,
    POSE,
    META_DATA,      // 推理结果等元数据
    TENSOR,          // 推理输入/输出 Tensor
    STREAM_END,
};

/**
 * @brief 基础数据包结构
 * 
 * 所有在管道中流动的数据都继承自此结构。
 * 使用 std::dynamic_pointer_cast 进行安全的类型转换。
 */
struct BasePacket {
    PacketType type = PacketType::UNKNOWN;
    int64_t timestamp_ms = 0;          // 毫秒时间戳
    uint32_t stream_id = 0;            // 多流并行的关键：用于区分不同的源流
    std::string source_id;             // 可选的源标识符
    int64_t frame_id;
    uint64_t cost_ms = 0;
    std::map<std::string,uint64_t> cost_time_map;

    virtual ~BasePacket() = default;
};

/**
 * @brief 原始视频包（编码数据）
 * 
 * 例如从 RTSP 读出的 H264/H265 NALU 单元
 */
struct RawVideoPacket : public BasePacket {
    RawVideoPacket() { type = PacketType::RAW_VIDEO; }
    
    std::vector<uint8_t> data;          // 编码数据缓冲区
    bool is_key_frame = false;          // 是否为关键帧
    int codec_id = 0;                   // 编码格式 ID（如 AV_CODEC_ID_H264）
};

/**
 * @brief 解码后的视频帧包
 *
 * 支持GPU内存零拷贝和CPU内存两种模式
 * 使用GPU内存时，避免CPU→GPU的数据复制
 */
struct VideoFramePacket : public BasePacket {
    VideoFramePacket() { type = PacketType::DECODED_FRAME; }

    // GPU内存支持（零拷贝模式）
    void* gpu_data = nullptr;           // GPU内存地址
    size_t gpu_size = 0;                // GPU内存大小
    bool is_gpu_memory = false;         // 是否为GPU内存模式

    // CPU内存（保留用于fallback）
    std::shared_ptr<cv::Mat> source_mat;
    std::shared_ptr<cv::Mat> mat;       // 图像数据（通常为 BGR 格式）

    int width = 0;
    int height = 0;
    int channels = 3;
};

/**
 * @brief 推理结果元数据包
 * 
 * 包含检测框、分类结果等信息
 */
struct InferenceResultPacket : public BasePacket {
    InferenceResultPacket() { type = PacketType::META_DATA; }
    struct KeyPoint {
        float x;
        float y;
        float confidence;
        bool visible;
        KeyPoint() : x(0.0f), y(0.0f), confidence(0.0f), visible(false) {}
        KeyPoint(float x_, float y_, float conf_, bool visible_)
            : x(x_), y(y_), confidence(conf_), visible(visible_) {}
    };
    /**
     * @brief 边界框结构
     */
    struct BBox {
        float x, y, w, h;               // 边界框坐标和尺寸
        float confidence;               // 置信度 [0.0, 1.0]
        int class_id;                   // 类别 ID
        std::string class_name;         // 类别名称
        bool has_keypoints;
        std::array<KeyPoint, 17> keypoints;
        float keypoints_conf;

        int track_id;
        int track_age;
        bool track_active;
        float smooth_x;
        float smooth_y;
        float smooth_w;
        float smooth_h;
        
        BBox() = default;
        BBox(float x_, float y_, float w_, float h_, float conf, int cls_id, const std::string& cls_name = "")
            : x(x_), y(y_), w(w_), h(h_), confidence(conf), class_id(cls_id), class_name(cls_name),
                track_id(-1), track_age(0), track_active(false), smooth_x(0.0f), smooth_y(0.0f), smooth_w(0.0f), smooth_h(0.0f) {}
    };

    struct PoseResult{
        float person_score;
        std::array<KeyPoint, 17> keypoints;
        cv::Rect2f person_box;
        int matched_det_idx;
        PoseResult() : person_score(0.0f), matched_det_idx(-1) {}
    };

    std::vector<BBox> detections;                       // 检测结果列表
    std::vector<PoseResult> pose_results;
    std::shared_ptr<VideoFramePacket> source_frame;     // 关联的原始帧，用于后续画框等操作
};

} // namespace core
} // namespace ai_stream