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
#include <set>

using namespace ai_stream;

// ========== 数据结构 ==========

struct FightingEvent {
    std::string type;           // "punch" | "fall" | "interaction"
    int track_id = -1;          // 当事人
    float confidence = 0.0f;    // 置信度
    std::vector<int> involved_track_ids;  // interaction双方
    std::string contact_parts;  // "wrist" | "head" | "near"
    int64_t timestamp_ms = 0;   // 事件发生时间（帧计数）
};

struct FightingResult {
    bool is_fighting = false;
    float fight_score = 0.0f;
    std::vector<FightingEvent> events;
    std::vector<int> active_track_ids;
    std::string current_state;              // 当前状态机状态字符串
    int fighting_duration_frames = 0;       // 当前状态持续帧数
    int total_fighting_frames = 0;          // 本次冲突累计帧数
};

// ========== 状态机定义 ==========

enum class FightingState {
    IDLE,           // 空闲/正常监控状态
    SUSPICIOUS,     // 可疑状态（检测到异常但不足够）
    CONFLICT,       // 冲突状态（确认打架中）
    COOLDOWN        // 冷却状态（打架结束后防止误触发）
};

// 状态转换事件
enum class StateTransitionEvent {
    NO_EVENT,                // 无异常事件
    INDIVIDUAL_SUSPICIOUS,   // 个体可疑（单人有异常动作）
    MULTI_PERSON_SUSPICIOUS, // 多人同时可疑
    CONFIRMED_FIGHTING,      // 确认打架（满足所有条件）
};

// ========== 主类 ==========

class FightingDetector {
public:
    struct Config {
        // 基础参数
        int history_frames = 10;
        float fps = 30.0f;                         // 帧率，用于速度归一化

        // Punch检测
        float punch_speed_threshold = 80.0f;        // 挥拳速度阈值 (pixels/frame)
        float punch_min_arm_extension = 0.4f;
        float punch_max_angle_diff = 45.0f;
        float punch_min_speed_drop_ratio = 0.7f;
        float punch_torso_angle_max = 60.0f;
        float punch_straight_angle_max = 45.0f;
        float punch_target_dist_max = 3.0f;
        float punch_target_dist_min = 0.5f;

        // Fall检测
        float fall_aspect_ratio_threshold = 2.0f;
        int fall_min_frames = 3;
        float fall_body_height_ratio = 0.35f;
        float fall_torso_angle_threshold = 30.0f;

        // Interaction检测
        float interaction_distance_threshold = 60.0f;
        float interaction_near_multiplier = 1.2f;
        float interaction_far_multiplier = 0.6f;
        float interaction_min_relative_speed = 30.0f;
        float interaction_approach_dot_threshold = -0.3f;

        // 状态机参数
        int suspicious_enter_threshold = 3;     // 进入SUSPICIOUS所需连续帧数
        int suspicious_exit_threshold = 5;        // 退出SUSPICIOUS所需连续正常帧数
        int conflict_enter_threshold = 5;         // 进入CONFLICT所需连续帧数
        int conflict_exit_threshold = 3;          // 退出CONFLICT所需连续正常帧数
        int cooldown_frames = 30;                 // 冷却帧数

        // 综合判定
        float fight_score_threshold = 0.5f;
        int min_involved_persons = 2;

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

    // 状态查询接口
    FightingState getCurrentState() const;
    std::string getCurrentStateString() const;
    int getStateDuration() const;
    int getTotalFightingFrames() const;

private:
    // ========== 状态机核心 ==========

    struct StateMachine {
        FightingState current_state = FightingState::IDLE;
        FightingState previous_state = FightingState::IDLE;
        int state_duration = 0;           // 当前状态持续帧数
        int total_fighting_frames = 0;    // 本次冲突累计帧数（CONFLICT状态累计）
        int suspicious_count = 0;         // 可疑计数
        int normal_count = 0;             // 正常计数
        int conflict_count = 0;           // 冲突计数
        int cooldown_count = 0;           // 冷却计数

        void reset();
        void update(StateTransitionEvent event, const Config& cfg);
        std::string getStateString() const;
    };

    // ========== 个体状态跟踪 ==========

    struct TrackState {
        std::deque<cv::Point2f> left_wrist_history;
        std::deque<cv::Point2f> right_wrist_history;
        std::deque<cv::Point2f> center_history;
        int punch_frame_count = 0;
        int fall_frame_count = 0;
        cv::Rect2f last_bbox;

        void reset();
    };

    // ========== 帧级分析结果 ==========

    struct FrameAnalysis {
        std::unordered_map<int, float> individual_scores;
        std::unordered_map<int, std::vector<FightingEvent>> individual_events;
        std::unordered_map<int, bool> is_suspicious;
        int suspicious_person_count = 0;
        int total_fall_count = 0;
        float total_score = 0.0f;
        bool has_interaction = false;

        void clear();
    };

    Config cfg_;
    StateMachine state_machine_;
    std::unordered_map<int, TrackState> tracks_;
    std::deque<bool> fight_history_;
    mutable std::mutex mutex_;
    static constexpr int MAX_HISTORY = 30;
    int64_t frame_counter_ = 0;

    // ========== 状态机驱动流程 ==========

    FrameAnalysis analyzeFrame(
        const std::vector<core::InferenceResultPacket::BBox>& detections);

    StateTransitionEvent determineTransitionEvent(
        const FrameAnalysis& analysis);

    FightingResult buildResult(
        const FrameAnalysis& analysis,
        const std::vector<int>& active_ids);

    // ========== 检测函数==========

    bool detectPunch(const core::InferenceResultPacket::BBox& det, 
                     TrackState& state, 
                     float& out_confidence,
                     const std::vector<core::InferenceResultPacket::BBox>& all_dets);

    bool detectFall(const core::InferenceResultPacket::BBox& det, TrackState& state);

    std::vector<FightingEvent> detectInteractions(
        const std::vector<core::InferenceResultPacket::BBox>& detections);

    // ========== 辅助函数==========

    static float calculateSpeed(const std::deque<cv::Point2f>& history);
    static float calculateSpeed(const std::deque<cv::Point2f>& history, int recent_n, int offset);
    static float calculateAngleChange(const std::deque<cv::Point2f>& history);
    static float euclideanDistance(const core::InferenceResultPacket::KeyPoint& a, 
                                   const core::InferenceResultPacket::KeyPoint& b);
    static bool isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp);

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

    void onEnterIdle();
    void onEnterSuspicious();
    void onEnterConflict();
    void onEnterCooldown();
    void onStateTick(FightingState state);
};