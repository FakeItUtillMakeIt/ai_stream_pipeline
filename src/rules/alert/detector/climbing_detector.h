// src/rules/alert/detector/climbing_detector.h

#pragma once

#include "ai_stream/core/packet.h"
#include "svm_model.h"

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
    std::string type;
    int track_id = -1;
    float confidence = 0.0f;
    std::vector<int> involved_track_ids;
    std::string detail;
    int64_t timestamp_ms = 0;
};

struct ClimbingResult {
    bool is_climbing = false;
    float climb_score = 0.0f;
    std::vector<ClimbingEvent> events;
    std::vector<int> active_track_ids;
    std::string current_state;
    int climbing_duration_frames = 0;
    int total_climbing_frames = 0;
};

// ========== 状态机定义 ==========

enum class ClimbingState {
    IDLE,
    SUSPICIOUS,
    CLIMBING,
    COOLDOWN
};

enum class StateTransitionEvent {
    NO_EVENT,
    INDIVIDUAL_SUSPICIOUS,
    MULTI_PERSON_SUSPICIOUS,
    CONFIRMED_CLIMBING,
};

// ========== 主类 ==========

class ClimbingDetector {
public:
    struct Config {
        int history_frames = 15;
        float fps = 30.0f;

        // === 静态骨架：手高于肩 ===
        float arm_raise_offset_ratio = 0.025f;

        // === 静态骨架：手臂弯曲 ===
        float arm_bend_min = 45.0f;
        float arm_bend_max = 155.0f;
        float arm_bend_ideal = 90.0f;

        // === 静态骨架：膝盖抬起 ===
        float knee_bend_min = 45.0f;
        float knee_bend_max = 155.0f;
        float knee_bend_ideal = 90.0f;

        // === 静态骨架：重心抬高/蜷缩 ===
        float center_raise_ratio = 0.06f;
        float stretch_compress_ratio = 0.8f;

        // === 身体朝向：倾斜角 ===
        float tilt_min = 5.0f;
        float tilt_max = 85.0f;

        // === 身体朝向：四肢张开 ===
        float limb_span_threshold = 0.5f;

        // === 动态特征：交替抬手抬脚 ===
        float alternation_ratio_threshold = 0.6f;
        int alternation_window = 10;
        float alternation_dx_threshold = 2.0f;

        // === 动态特征：整体向上移动 ===
        float ascent_slope_ratio = -0.002f; 
        int ascent_min_frames = 5;
        float net_displacement_ratio = 0.02f; //移动速度快的话需要调大

        // === 过滤器 ===
        float oscillation_threshold_high = 0.7f;
        float oscillation_threshold_low = 0.3f;
        float lateral_threshold_high = 0.6f;
        float lateral_threshold_low = 0.3f;
        float burst_threshold_high = 0.5f;
        float burst_threshold_low = 0.3f;

        // === 状态机 ===
        int suspicious_enter_threshold = 3;
        int suspicious_exit_threshold = 5;
        int climbing_enter_threshold = 3;
        int climbing_exit_threshold = 5;
        int cooldown_frames = 30;

        // === 综合判定 ===
        float climb_score_threshold = 0.4f;
        int min_involved_persons = 1;

        // === ML 判定模式 ===
        bool use_ml_score = false;
        std::string ml_model_path;
        std::string ml_scaler_path;
        std::string training_data_path="./feature_data.csv";
        std::string video_id = "default";

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

    ClimbingState getCurrentState() const;
    std::string getCurrentStateString() const;
    int getStateDuration() const;
    int getTotalClimbingFrames() const;

private:
    // ========== 状态机 ==========
    struct StateMachine {
        ClimbingState current_state = ClimbingState::IDLE;
        ClimbingState previous_state = ClimbingState::IDLE;
        int state_duration = 0;
        int total_climbing_frames = 0;
        int suspicious_count = 0;
        int normal_count = 0;
        int climbing_count = 0;
        int cooldown_count = 0;

        void reset();
        void update(StateTransitionEvent event, const Config& cfg);
        std::string getStateString() const;
    };

    // ========== 个体状态跟踪 ==========
    struct TrackState {
        std::deque<cv::Point2f> center_history;
        std::deque<float> left_wrist_y_history;
        std::deque<float> right_wrist_y_history;
        std::deque<float> left_wrist_x_history;
        std::deque<float> right_wrist_x_history;
        std::deque<float> left_ankle_y_history;
        std::deque<float> right_ankle_y_history;
        float avg_center_y = -1.0f;
        int posture_frame_count = 0;
        int ascent_frame_count = 0;
        cv::Rect2f last_bbox;

        std::vector<float> feature_vector;
        bool feature_valid = false;

        void reset();
    };

    // ========== 帧级分析结果 ==========
    struct FrameAnalysis {
        std::unordered_map<int, float> individual_scores;
        std::unordered_map<int, std::vector<ClimbingEvent>> individual_events;
        std::unordered_map<int, bool> is_suspicious;
        int suspicious_person_count = 0;
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

    LinearSVMModel svm_model_;
    bool svm_loaded_ = false;

    // ========== 状态机驱动 ==========
    FrameAnalysis analyzeFrame(
        const std::vector<core::InferenceResultPacket::BBox>& detections);
    StateTransitionEvent determineTransitionEvent(
        const FrameAnalysis& analysis);
    ClimbingResult buildResult(
        const FrameAnalysis& analysis,
        const std::vector<int>& active_ids);

    // ========== 辅助函数 ==========
    static float calculateAngle(
        const core::InferenceResultPacket::KeyPoint& a,
        const core::InferenceResultPacket::KeyPoint& b,
        const core::InferenceResultPacket::KeyPoint& c);
    static float calculateOscillation(const std::deque<float>& y_history);
    static float calculateLateralMovement(const std::deque<cv::Point2f>& center_history);
    static float calculateMovementBurst(const std::deque<cv::Point2f>& center_history);
    static bool isKeypointValid(const core::InferenceResultPacket::KeyPoint& kp);
    static float euclideanDistance(
        const core::InferenceResultPacket::KeyPoint& a,
        const core::InferenceResultPacket::KeyPoint& b);
    static float euclideanDistance(const cv::Point2f& a, const cv::Point2f& b);
    void cleanupInactive(const std::vector<int>& active_ids);

    // ========== 静态特征检测 ==========
    bool detectHandAboveShoulder(
        const core::InferenceResultPacket::BBox& det,
        float& out_conf);
    bool detectArmBend(
        const core::InferenceResultPacket::BBox& det,
        float& out_conf);
    bool detectKneeRaise(
        const core::InferenceResultPacket::BBox& det,
        float& out_conf);
    bool detectCenterRaise(
        const core::InferenceResultPacket::BBox& det,
        TrackState& state,
        float& out_conf);
    bool detectBodyTilt(
        const core::InferenceResultPacket::BBox& det,
        float& out_conf);
    bool detectLimbSpan(
        const core::InferenceResultPacket::BBox& det,
        float& out_conf);

    // ========== 动态特征检测 ==========
    bool detectAlternatingLimb(
        const core::InferenceResultPacket::BBox& det,
        TrackState& state,
        float& out_conf);
    bool detectOverallAscent(
        const core::InferenceResultPacket::BBox& det,
        TrackState& state,
        float& out_conf);

    // ========== 过滤器 ==========
    float filterByOscillation(const TrackState& state) const;
    float filterByLateralMovement(const TrackState& state) const;
    float filterByMovementBurst(const TrackState& state) const;

    float computeMLScore(const TrackState& state) const;
    void collectFeatureVector(TrackState& state, float hand_conf, float arm_conf,
                              float knee_conf, float center_conf, float tilt_conf,
                              float limb_conf, float alt_conf, float ascent_conf,
                              bool has_ascent, float osc, float lat, float burst);
    void exportTrainingData(const std::unordered_map<int, TrackState>& tracks,
                            const FrameAnalysis& analysis, int64_t frame_id);

    // ========== 状态机回调 ==========
    void onEnterIdle();
    void onEnterSuspicious();
    void onEnterClimbing();
    void onEnterCooldown();
    void onStateTick(ClimbingState state);
};
