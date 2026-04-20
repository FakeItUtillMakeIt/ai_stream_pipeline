// src/nodes/track/track_result.h
#pragma once

#include <Eigen/Dense>
#include <vector>

namespace ai_stream {
namespace nodes {

/**
 * @brief 统一的跟踪结果结构
 */
struct UnifiedTrackResult {
    float x, y, w, h;           // 边界框
    float confidence;           // 置信度
    int class_id;               // 类别 ID
    int track_id;               // 跟踪 ID
    int age;                    // 跟踪帧数
    bool active;                // 是否活跃
    
    // 卡尔曼滤波状态（可选）
    float smooth_x, smooth_y, smooth_w, smooth_h;
};

} // namespace nodes
} // namespace ai_stream