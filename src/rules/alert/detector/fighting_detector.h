// src/rules/alert/detector/fighting_detector.h

#pragma once

#include "ai_stream/core/packet.h"

#include <opencv2/opencv.hpp>
#include <vector>
#include <deque>
#include <unordered_map>
#include <string>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace ai_stream;

struct FightingEvent {
    std::string type;           // "punch" | "fall" | "interaction"
    int track_id = -1;          // 当事人
    float confidence = 0.0f;    // punch置信度
    std::vector<int> involved_track_ids;  // interaction双方
    std::string contact_parts;  // "wrist" | "head" | "near"
};

struct FightingResult {
    bool is_fighting = false;
    float fight_score = 0.0f;
    std::vector<FightingEvent> events;
    std::vector<int> active_track_ids;  // 当前活跃的所有track_id
};

// ========== 主类 ==========

class FightingDetector {
public:
    struct Config {
        // Punch检测
        int history_frames = 10;
        float punch_speed_threshold = 25.0f;

        // Fall检测
        float fall_aspect_ratio_threshold = 1.2f;

        // Interaction检测
        float interaction_distance_threshold = 80.0f;
        float interaction_near_multiplier = 1.5f;

        // 综合判定
        int fight_frames_threshold = 5;
        float fight_score_threshold = 0.5f;

        // 关键点索引 (COCO)
        static constexpr int NOSE = 0;
        static constexpr int LEFT_WRIST = 9;
        static constexpr int RIGHT_WRIST = 10;
    };

    explicit FightingDetector(const Config& cfg);
    ~FightingDetector() = default;

    // 处理单帧，输入该帧所有检测框（必须含track_id和keypoints）
    FightingResult process(const std::vector<core::InferenceResultPacket::BBox>& detections);

    // 重置所有状态（切换视频流时调用）
    void reset();

private:
    // ========== 子检测器状态 ==========
    struct TrackState {
        // Punch
        std::deque<cv::Point2f> left_wrist_history;
        std::deque<cv::Point2f> right_wrist_history;
        int punch_frame_count = 0;

        // Fall
        int fall_frame_count = 0;

        // 最新bbox
        cv::Rect2f last_bbox;
    };

    Config cfg_;
    std::unordered_map<int, TrackState> tracks_;   // track_id -> state
    std::unordered_map<int, int> fighting_tracks_; // track_id -> 连续异常帧数
    std::deque<bool> fight_history_;               // 历史判定结果（用于时序平滑）
    static constexpr int MAX_HISTORY = 30;

    // ========== 内部检测函数 ==========
    bool detectPunch(const core::InferenceResultPacket::BBox& det, TrackState& state, float& out_confidence);
    bool detectFall(const core::InferenceResultPacket::BBox& det, TrackState& state);
    std::vector<FightingEvent> detectInteractions(const std::vector<core::InferenceResultPacket::BBox>& detections);

    // 辅助函数
    static float calculateSpeed(const std::deque<cv::Point2f>& history);
    static float calculateAngleChange(const std::deque<cv::Point2f>& history);
    static float euclideanDistance(const core::InferenceResultPacket::KeyPoint& a, const core::InferenceResultPacket::KeyPoint& b);
    static bool isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp);
    void cleanupInactive(const std::vector<int>& active_ids);
};

