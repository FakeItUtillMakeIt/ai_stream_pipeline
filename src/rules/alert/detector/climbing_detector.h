// src/rules/alert/detector/climbing_detector.h

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

// ========== 数据结构 ==========

struct ClimbingEvent {
    std::string type;           // "posture" | "ascent" | "limb_asymmetry" | "foot_off_ground"
    int track_id = -1;          // 当事人
    float confidence = 0.0f;    // 置信度
    std::vector<int> involved_track_ids;
    std::string detail;         // 详细描述
    int64_t timestamp_ms = 0;   // 事件发生时间（帧计数）
};

struct ClimbingResult {
    bool is_climbing = false;
    float climb_score = 0.0f;
    std::vector<ClimbingEvent> events;
    std::vector<int> active_track_ids;
    std::string current_state;              // 当前状态机状态字符串
    int climbing_duration_frames = 0;         // 当前状态持续帧数
    int total_climbing_frames = 0;          // 本次攀爬累计帧数
};

// ========== 状态机定义 ==========

enum class ClimbingState {
    IDLE,           // 空闲/正常监控状态
    SUSPICIOUS,     // 可疑状态（检测到异常姿态但不足够）
    CLIMBING,       // 攀爬状态（确认攀爬中）
    COOLDOWN        // 冷却状态（攀爬结束后防止误触发）
};

// 状态转换事件
enum class StateTransitionEvent {
    NO_EVENT,                // 无异常事件
    INDIVIDUAL_SUSPICIOUS,   // 个体可疑（单人有异常姿态/上升）
    MULTI_PERSON_SUSPICIOUS, // 多人同时可疑或强特征
    CONFIRMED_CLIMBING,      // 确认攀爬（满足所有条件）
};

// ========== 主类 ==========

class ClimbingDetector {
public:
    struct Config {
        // 基础参数
        int history_frames = 10;
        float fps = 30.0f;

        // 姿态检测
        float torso_angle_climb_min = 45.0f;    // 躯干角度偏离垂直的最小值（度）
        float torso_angle_climb_max = 135.0f;   // 躯干角度偏离垂直的最大值
        float arm_raise_threshold = 0.0f;         // 手腕高于肩膀的像素阈值（>0表示必须高于）
        float leg_raise_threshold = 0.0f;       // 脚踝高于膝盖的像素阈值
        float body_stretch_ratio = 1.3f;        // 身体拉伸比例（头-踝距离 / bbox高度）
        float body_compress_ratio = 0.6f;       // 身体压缩比例（头-踝距离 / bbox高度）
        float aspect_ratio_threshold = 1.2f;    // 宽高比阈值（w/h > 阈值认为水平姿态/翻越）

        // 垂直上升检测
        float ascent_speed_threshold = 2.0f;    // 向上速度阈值（像素/帧，Y轴向下，故为负速度阈值）
        int ascent_min_frames = 5;              // 最小持续上升帧数
        float ascent_min_distance = 20.0f;      // 最小上升距离（像素）

        // 肢体不对称（攀爬交替特征）
        float limb_asymmetry_threshold = 0.15f;   // 左右肢体高度差占bbox高度的比例

        // 脚部离地
        float foot_off_ground_ratio = 0.3f;     // 脚踝位于bbox上半部分的比例阈值

        // 状态机参数
        int suspicious_enter_threshold = 3;     // 进入SUSPICIOUS所需连续可疑帧数
        int suspicious_exit_threshold = 5;      // 退出SUSPICIOUS所需连续正常帧数
        int climbing_enter_threshold = 1;       // 进入CLIMBING所需连续确认帧数
        int climbing_exit_threshold = 5;          // 退出CLIMBING所需连续正常帧数
        int cooldown_frames = 30;                 // 冷却帧数

        // 综合判定
        float climb_score_threshold = 0.5f;
        int min_involved_persons = 1;           // 攀爬通常是单人行为

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

    explicit ClimbingDetector(const Config& cfg);
    ~ClimbingDetector() = default;

    ClimbingResult process(const std::vector<core::InferenceResultPacket::BBox>& detections);
    void reset();

    // 状态查询接口
    ClimbingState getCurrentState() const;
    std::string getCurrentStateString() const;
    int getStateDuration() const;
    int getTotalClimbingFrames() const;

private:
    // ========== 状态机核心 ==========

    struct StateMachine {
        ClimbingState current_state = ClimbingState::IDLE;
        ClimbingState previous_state = ClimbingState::IDLE;
        int state_duration = 0;           // 当前状态持续帧数
        int total_climbing_frames = 0;    // 本次攀爬累计帧数（CLIMBING状态累计）
        int suspicious_count = 0;         // 可疑计数
        int normal_count = 0;             // 正常计数
        int climbing_count = 0;           // 攀爬计数
        int cooldown_count = 0;           // 冷却计数

        void reset();
        void update(StateTransitionEvent event, const Config& cfg);
        std::string getStateString() const;
    };

    // ========== 个体状态跟踪 ==========

    struct TrackState {
        std::deque<cv::Point2f> center_history;
        std::deque<float> left_wrist_y_history;
        std::deque<float> right_wrist_y_history;
        std::deque<float> left_ankle_y_history;
        std::deque<float> right_ankle_y_history;
        int posture_frame_count = 0;
        int ascent_frame_count = 0;
        cv::Rect2f last_bbox;

        void reset();
    };

    // ========== 帧级分析结果 ==========

    struct FrameAnalysis {
        std::unordered_map<int, float> individual_scores;
        std::unordered_map<int, std::vector<ClimbingEvent>> individual_events;
        std::unordered_map<int, bool> is_suspicious;
        int suspicious_person_count = 0;
        int total_climbing_person_count = 0;
        float total_score = 0.0f;
        bool has_ascent = false;

        void clear();
    };

    Config cfg_;
    StateMachine state_machine_;
    std::unordered_map<int, TrackState> tracks_;
    std::deque<bool> climb_history_;
    mutable std::mutex mutex_;
    static constexpr int MAX_HISTORY = 30;
    int64_t frame_counter_ = 0;

    // ========== 状态机驱动流程 ==========

    FrameAnalysis analyzeFrame(
        const std::vector<core::InferenceResultPacket::BBox>& detections);

    StateTransitionEvent determineTransitionEvent(
        const FrameAnalysis& analysis);

    ClimbingResult buildResult(
        const FrameAnalysis& analysis,
        const std::vector<int>& active_ids);

    // ========== 检测函数 ==========

    bool detectClimbingPosture(const core::InferenceResultPacket::BBox& det,
                               TrackState& state,
                               float& out_confidence);

    bool detectVerticalAscent(const core::InferenceResultPacket::BBox& det,
                              TrackState& state,
                              float& out_confidence);

    bool detectLimbAsymmetry(const core::InferenceResultPacket::BBox& det,
                             TrackState& state,
                             float& out_confidence);

    bool detectFootOffGround(const core::InferenceResultPacket::BBox& det,
                             float& out_confidence);

    // ========== 辅助函数 ==========

    static float calculateSpeedY(const std::deque<cv::Point2f>& history);
    static float calculateTorsoAngle(const std::array<core::InferenceResultPacket::KeyPoint, 17>& kpts);
    static bool isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp);
    static float euclideanDistance(const core::InferenceResultPacket::KeyPoint& a,
                                   const core::InferenceResultPacket::KeyPoint& b);
    static float normalizeAngle(float angle);

    void cleanupInactive(const std::vector<int>& active_ids);

    void onEnterIdle();
    void onEnterSuspicious();
    void onEnterClimbing();
    void onEnterCooldown();
    void onStateTick(ClimbingState state);
};