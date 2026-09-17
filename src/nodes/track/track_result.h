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
    float x = 0, y = 0, w = 0, h = 0;   // 边界框
    float confidence = 0;               // 置信度
    int class_id = -1;                  // 类别 ID
    int track_id = -1;                  // 跟踪 ID
    int age = 0;                        // 跟踪帧数
    bool active = false;                // 是否活跃

    // 卡尔曼滤波状态（可选）
    float smooth_x = 0, smooth_y = 0, smooth_w = 0, smooth_h = 0;
};

} // namespace nodes
} // namespace ai_stream