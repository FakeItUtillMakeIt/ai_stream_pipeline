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

using namespace ai_stream;

struct RisingTrackState {
    std::deque<double> y_history;                   // y坐标历史
    std::deque<double> x_history;                   // x坐标历史
    std::deque<double> h_history;                   // 人体框高度历史
    std::deque<double> w_history;                   // 人体框宽度历史
    std::deque<double> speed_history;               // 速度历史
    std::deque<double> acceleration_history;        // 加速度历史
    std::deque<double> smoothed_y_history;          // 平滑后的y坐标历史
    bool is_rising;                                 // 是否上升
    int rising_frames;                              // 连续上升帧计数
    int non_rising_frames;                          // 连续非上升帧计数
    int frame_count;                                // 帧计数器
    double rising_score;                            // 当前上升评分
    
    RisingTrackState() : is_rising(false), rising_frames(0), non_rising_frames(0), frame_count(0), rising_score(0.0) {}
};

struct RisingResult {
    bool is_rising;                                 // 是否有任何人上升
    std::unordered_map<int, bool> track_rising_states;  // 每个track_id的上升状态
    std::unordered_map<int, double> track_rising_scores; // 每个track_id的上升评分
    std::vector<int> active_track_ids;              // 活跃的track_id列表
};

class PersonBoxRisingDetector {
public:
    explicit PersonBoxRisingDetector(const std::string& video_path = "");
    
    RisingResult process(const std::vector<core::InferenceResultPacket::BBox>& detections);
    
    bool get_rising_state(int track_id) const;
    
    void cleanup_old_tracks(const std::unordered_set<int>& active_track_ids, int max_age = 50);
    
private:
    std::string video_path_;
    std::unordered_map<int, RisingTrackState> track_rising_state_;
    
    double rising_speed_threshold_;      // 上升速度阈值（像素/帧，负值表示上升）
    double non_rising_speed_threshold_;  // 非上升速度阈值
    int window_size_;                    // 滑动窗口大小
    int history_length_;                 // 保留最近N帧的历史
    int required_rising_frames_;         // 需要连续多少帧才标记为上升
    double max_x_displacement_;          // X轴最大位移阈值（超过则不判定为上升）
    double min_height_ratio_;            // 最小高度比例（框变小太多则不判定为上升）
    double max_oscillation_range_;       // 最大振荡范围（超过则认为是跳舞等动作）
    double min_sustain_ratio_;           // 最小持续上升比例（上升帧数/总帧数）
    int min_camera_shake_tracks_;        // 最少多少人同时上升才判定为镜头晃动
    double max_acceleration_;            // 最大加速度阈值（超过则认为是突然动作）
    double aspect_ratio_change_threshold_; // 宽高比变化阈值（超过则认为是蹲下/起身等动作）
    double smooth_factor_;               // 平滑因子（0-1，越大越平滑）
    double min_direction_consistency_;   // 最小方向一致性（同向变化帧数/总帧数）
    double min_rising_score_;            // 最小上升评分阈值
    int rising_falling_tolerance_;       // rising_frames递减容错帧数（允许短暂中断）
    double score_slope_weight_;          // 斜率评分权重
    double score_x_weight_;             // X位移评分权重
    double score_height_weight_;        // 高度稳定评分权重
    double score_oscillation_weight_;   // 振荡评分权重
    double score_acceleration_weight_;  // 加速度评分权重
    double score_aspect_ratio_weight_;  // 宽高比评分权重
    double score_direction_weight_;     // 方向一致性评分权重
    
    bool update_track(int track_id, const core::InferenceResultPacket::BBox& bbox);
    
    double calculate_y_center(const core::InferenceResultPacket::BBox& bbox) const;
    
    double calculate_x_center(const core::InferenceResultPacket::BBox& bbox) const;
    
    double calculate_slope(const std::deque<double>& data) const;
    
    bool check_x_displacement(const std::deque<double>& x_history) const;
    
    bool check_height_stability(const std::deque<double>& h_history) const;
    
    bool check_oscillation(const std::deque<double>& y_history) const;
    
    bool check_sustain_ratio(const RisingTrackState& state) const;

    bool check_sustain_ratio_recent(const RisingTrackState& state) const;

    bool check_camera_x_consistency(const std::vector<int>& rising_track_ids) const;
    
    bool check_acceleration(const std::deque<double>& acceleration_history) const;
    
    bool check_aspect_ratio_stability(const std::deque<double>& w_history, const std::deque<double>& h_history) const;
    
    bool check_direction_consistency(const std::deque<double>& y_history) const;
    
    std::deque<double> smooth_trajectory(const std::deque<double>& y_history) const;
    
    double calculate_rising_score(double trend_slope,
                                  bool x_displacement_ok,
                                  bool height_stable,
                                  bool no_oscillation,
                                  bool acceleration_ok,
                                  bool aspect_ratio_ok,
                                  bool direction_ok) const;
};

#endif // PERSON_BOX_RISING_DETECTOR_H
