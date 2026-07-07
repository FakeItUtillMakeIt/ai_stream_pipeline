#ifndef PERSON_BOX_RISING_DETECTOR_H
#define PERSON_BOX_RISING_DETECTOR_H

#include "utils/zone_utils.h"
#include "ai_stream/core/packet.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <deque>
#include <memory>
#include <unordered_set>
#include <mutex>

using namespace ai_stream;

// 攀爬状态枚举
enum class ClimbState {
    IDLE,           // 初始/静止状态
    TRACKING,       // 正在跟踪，但未确认攀爬
    CLIMBING,       // 确认攀爬中
    EXITED          // 退出攀爬状态
};

// 攀爬事件枚举（用于状态机转换）
enum class ClimbEvent {
    NO_SIGNAL,          // 无有效信号（跟踪帧数不足）
    RISING_DETECTED,    // 检测到上升趋势（分数达标但连续帧数不足）
    RISING_CONFIRMED,   // 连续多帧确认上升
    RISING_LOST,        // 上升趋势丢失（连续非上升帧数达标）
    TRACK_LOST          // 跟踪目标丢失/超时
};

struct RisingTrackState {
    // ========== 归一化坐标历史 (0.0 ~ 1.0) ==========
    std::deque<double> y_history_norm;          // 归一化Y中心
    std::deque<double> x_history_norm;          // 归一化X中心
    std::deque<double> h_history_norm;          // 归一化高度
    std::deque<double> w_history_norm;          // 归一化宽度
    std::deque<double> speed_history_norm;      // 归一化Y速度 (高度/帧)
    std::deque<double> acceleration_history_norm; // 归一化加速度
    std::deque<double> smoothed_y_history_norm; // 平滑后的归一化Y

    // ========== 原始像素值（仅用于日志/debug） ==========
    std::deque<double> y_pixel_history;
    std::deque<double> x_pixel_history;
    std::deque<double> h_pixel_history;
    std::deque<double> w_pixel_history;

    // ========== 状态机相关 ==========
    bool is_rising;
    int rising_frames;
    int non_rising_frames;
    int frame_count;
    int last_update_frame;
    int consecutive_tracking;


    // ========== 分数 ==========
    double rising_score;
    double smoothed_score;

    // ========== 状态机状态 ==========
    ClimbState current_state;
    int state_entry_frame;

    // ========== 视频尺寸 ==========
    int video_width;
    int video_height;

    RisingTrackState() 
        : is_rising(false)
        , rising_frames(0)
        , non_rising_frames(0)
        , frame_count(0)
        , last_update_frame(0)
        , consecutive_tracking(0)
        , rising_score(0.0)
        , smoothed_score(0.0)
        , current_state(ClimbState::IDLE)
        , state_entry_frame(0)
        , video_width(0)
        , video_height(0)
    {}
};

struct RisingResult {
    int frame_id;
    bool is_rising;
    std::unordered_map<int, bool> track_rising_states;
    std::unordered_map<int, double> track_rising_scores;
    std::vector<int> active_track_ids;
    std::unordered_map<int, ClimbState> track_climb_states;
    bool is_camera_shake;

    RisingResult() : frame_id(0), is_rising(false), is_camera_shake(false) {}
};

class PersonBoxRisingDetector {
public:
    explicit PersonBoxRisingDetector(const std::string& video_path = "");

    // 批量处理接口
    RisingResult process(const std::vector<core::InferenceResultPacket::BBox>& detections, int frame_id);

    bool get_rising_state(int track_id) const;
    ClimbState get_climb_state(int track_id) const;
    std::string get_climb_state_string(int track_id) const;

    void cleanup_old_tracks(const std::unordered_set<int>& active_track_ids, int max_age = 50);

    // 设置视频分辨率（用于归一化）
    void set_video_resolution(int width, int height);
    std::pair<int, int> get_video_resolution() const;

private:
    mutable std::mutex mutex_;
    std::string video_path_;
    std::unordered_map<int, RisingTrackState> track_rising_state_;

    int video_width_;
    int video_height_;
    bool resolution_set_;

    // ========== 归一化阈值（与分辨率无关） ==========
    // 速度阈值：归一化Y速度 < -0.015 (每帧向上移动1.5%画面高度)
    double rising_speed_threshold_;
    int window_size_;
    int history_length_;
    int required_rising_frames_;
    int required_non_rising_frames_;
    int min_tracking_frames_;

    // 空间稳定性（归一化值）
    double max_x_displacement_;         // 最大X位移（归一化）
    double min_height_ratio_;           // 最小高度比例
    double max_oscillation_range_;      // 最大振荡范围（归一化）

    // 动态阈值
    double max_acceleration_;           // 最大归一化加速度
    double aspect_ratio_change_threshold_;
    double smooth_factor_;
    double min_direction_consistency_;

    // 分数阈值
    double min_rising_score_;
    double score_smoothing_factor_;

    // 分数权重（总和为1.0）
    double score_slope_weight_;
    double score_x_weight_;
    double score_height_weight_;
    double score_oscillation_weight_;
    double score_acceleration_weight_;
    double score_aspect_ratio_weight_;
    double score_direction_weight_;

    // 镜头晃动
    int min_camera_shake_tracks_;

    // ========== 归一化辅助函数 ==========
    double normalize_x(double x_pixel) const;
    double normalize_y(double y_pixel) const;
    double normalize_w(double w_pixel) const;
    double normalize_h(double h_pixel) const;
    double denormalize_y(double y_norm) const;
    double denormalize_h(double h_norm) const;

    // ========== 核心计算 ==========
    double calculate_y_center_pixel(const core::InferenceResultPacket::BBox& bbox) const;
    double calculate_x_center_pixel(const core::InferenceResultPacket::BBox& bbox) const;
    double calculate_normalized_speed(double y_pixel_current, double y_pixel_prev) const;
    double calculate_slope(const std::deque<double>& data) const;

    // ========== 检查函数（全部基于归一化坐标） ==========
    bool check_x_displacement(const std::deque<double>& x_history_norm) const;
    bool check_height_stability(const std::deque<double>& h_history_norm) const;
    bool check_oscillation(const std::deque<double>& y_history_norm) const;
    bool check_sustain_ratio_recent(const RisingTrackState& state) const;
    bool check_acceleration(const std::deque<double>& acceleration_history_norm) const;
    bool check_aspect_ratio_stability(const std::deque<double>& w_history_norm, 
                                       const std::deque<double>& h_history_norm) const;
    bool check_direction_consistency(const std::deque<double>& y_history_norm) const;

    // 镜头晃动检测（基于归一化坐标）
    bool check_camera_shake(const std::vector<int>& rising_track_ids) const;
    bool check_camera_x_consistency(const std::vector<int>& rising_track_ids) const;
    bool check_camera_y_consistency(const std::vector<int>& rising_track_ids) const;

    std::deque<double> smooth_trajectory(const std::deque<double>& y_history_norm) const;

    // ========== 分数计算（归一化版本） ==========
    double calculate_rising_score(double trend_slope_norm,
                                  bool x_displacement_ok,
                                  bool height_stable,
                                  bool no_oscillation,
                                  bool acceleration_ok,
                                  bool aspect_ratio_ok,
                                  bool direction_ok,
                                  double sustain_ratio) const;

    // ========== 状态机 ==========
    ClimbEvent evaluate_event(const RisingTrackState& state, bool is_currently_rising) const;
    void transition_state(RisingTrackState& state, ClimbEvent event, int frame_id);
    ClimbState get_next_state(ClimbState current, ClimbEvent event) const;
    void on_enter_state(RisingTrackState& state, ClimbState new_state, int frame_id);
    void on_exit_state(RisingTrackState& state, ClimbState old_state, int frame_id);

    bool update_track(int track_id, const core::InferenceResultPacket::BBox& bbox, int frame_id);
};

#endif // PERSON_BOX_RISING_DETECTOR_H