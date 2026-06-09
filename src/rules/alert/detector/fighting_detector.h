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
#include <mutex>
#include <unordered_set>

using namespace ai_stream;

struct FightingEvent {
    std::string type;           // "punch" | "fall" | "interaction"
    int track_id = -1;          // 当事人
    float confidence = 0.0f;    // 置信度
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
        float punch_speed_threshold = 120.0f;
        float punch_min_arm_extension = 0.4f;   // 手臂伸展度最小比例
        float punch_max_angle_diff = 45.0f;     // 与目标方向最大偏差
        float punch_min_speed_drop_ratio = 0.7f; // 收拳速度下降比例
        
        // Fall检测
        float fall_aspect_ratio_threshold = 2.0f;
        int fall_min_frames = 3;                // 最少持续帧数
        float fall_body_height_ratio = 0.5f;    // 关键点高度占bbox比例
        float fall_torso_angle_threshold = 30.0f; // 躯干水平角度阈值
        
        // Interaction检测
        float interaction_distance_threshold = 60.0f;
        float interaction_near_multiplier = 1.2f;
        float interaction_min_relative_speed = 30.0f;
        float interaction_approach_dot_threshold = -0.3f; // 相向运动点积阈值
        
        // 综合判定
        int fight_frames_threshold = 5;
        float fight_score_threshold = 0.5f;
        int min_involved_persons = 2;           // 最少参与人数
        
        // 关键点索引 (COCO)
        static constexpr int NOSE = 0;
        static constexpr int LEFT_EYE = 1;
        static constexpr int RIGHT_EYE = 2;
        static constexpr int LEFT_EAR = 3;
        static constexpr int RIGHT_EAR = 4;
        static constexpr int LEFT_SHOULDER = 5;
        static constexpr int RIGHT_SHOULDER = 6;
        static constexpr int LEFT_ELBOW = 7;
        static constexpr int RIGHT_ELBOW = 8;
        static constexpr int LEFT_WRIST = 9;
        static constexpr int RIGHT_WRIST = 10;
        static constexpr int LEFT_HIP = 11;
        static constexpr int RIGHT_HIP = 12;
        static constexpr int LEFT_KNEE = 13;
        static constexpr int RIGHT_KNEE = 14;
        static constexpr int LEFT_ANKLE = 15;
        static constexpr int RIGHT_ANKLE = 16;
    };

    explicit FightingDetector(const Config& cfg);
    ~FightingDetector() = default;

    FightingResult process(const std::vector<core::InferenceResultPacket::BBox>& detections);
    void reset();

private:
    struct TrackState {
        std::deque<cv::Point2f> left_wrist_history;
        std::deque<cv::Point2f> right_wrist_history;
        std::deque<cv::Point2f> center_history;   // 新增：中心点历史（用于速度计算）
        int punch_frame_count = 0;
        int fall_frame_count = 0;
        cv::Rect2f last_bbox;
    };

    Config cfg_;
    std::unordered_map<int, TrackState> tracks_;
    std::unordered_map<int, int> fighting_tracks_;
    std::deque<bool> fight_history_;
    mutable std::mutex mutex_;
    static constexpr int MAX_HISTORY = 30;

    // 核心检测函数
    bool detectPunch(const core::InferenceResultPacket::BBox& det, 
                     TrackState& state, 
                     float& out_confidence,
                     const std::vector<core::InferenceResultPacket::BBox>& all_dets);
    bool detectFall(const core::InferenceResultPacket::BBox& det, TrackState& state);
    std::vector<FightingEvent> detectInteractions(
        const std::vector<core::InferenceResultPacket::BBox>& detections);

    // 辅助函数
    static float calculateSpeed(const std::deque<cv::Point2f>& history);
    static float calculateSpeed(const std::deque<cv::Point2f>& history, int recent_n, int offset);
    static float calculateAngleChange(const std::deque<cv::Point2f>& history);
    static float euclideanDistance(const core::InferenceResultPacket::KeyPoint& a, 
                                   const core::InferenceResultPacket::KeyPoint& b);
    static bool isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp);
    
    // 新增辅助函数
    static float calculateArmExtension(const core::InferenceResultPacket::KeyPoint& shoulder,
                                       const core::InferenceResultPacket::KeyPoint& elbow,
                                       const core::InferenceResultPacket::KeyPoint& wrist);
    static float calculateTorsoAngle(const std::array<core::InferenceResultPacket::KeyPoint, 17>& kpts);
    static cv::Point2f calculateVelocity(const TrackState& state);
    static bool checkFacingEachOther(const core::InferenceResultPacket::BBox& d1,
                                      const core::InferenceResultPacket::BBox& d2);
    static float normalizeAngle(float angle);
    int countFalls(const std::vector<core::InferenceResultPacket::BBox>& detections);
    
    void cleanupInactive(const std::vector<int>& active_ids);
};