#include "person_box_rising_detector.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>

PersonBoxRisingDetector::PersonBoxRisingDetector(const std::string& video_path)
    : video_path_(video_path)
    , rising_speed_threshold_(-2.0)  // 负值表示上升（Y轴向下为正），使用斜率检测缓慢上升
    , non_rising_speed_threshold_(2.0)
    , window_size_(10)
    , history_length_(30)
    , required_rising_frames_(5)     // 增加要求连续上升帧数，减少跳舞等周期性动作误判
    , max_x_displacement_(10.0)      // X轴位移超过10.0不判定为上升
    , min_height_ratio_(0.8)         // 人体框高度缩小到80%以下不判定为上升
    , max_oscillation_range_(30.0)   // Y轴振荡范围超过30像素认为是周期性动作
    , min_sustain_ratio_(0.6)        // 至少60%的帧需要在上升状态
    , min_camera_shake_tracks_(2)    // 至少2人同时上升判定为镜头晃动
    , max_acceleration_(15.0)        // 加速度超过15像素/帧²认为是突然动作
    , aspect_ratio_change_threshold_(0.3)  // 宽高比变化超过30%认为是蹲下/起身
    , smooth_factor_(0.6)            // 平滑因子，0.6表示当前帧权重0.6，减少拖尾滞后
    , min_direction_consistency_(0.6) // 至少60%的帧同向变化才认为是持续上升
    , min_rising_score_(0.7)         // 评分超过0.7判定为上升
    , rising_falling_tolerance_(2)  // 允许短暂中断帧数，超过2帧才复位rising_frames
    , score_slope_weight_(0.25)     // 斜率bonus权重
    , score_x_weight_(0.20)         // X位移惩罚权重
    , score_height_weight_(0.10)    // 高度稳定惩罚权重
    , score_oscillation_weight_(0.0) // 振荡已作为硬性否决，不再评分
    , score_acceleration_weight_(0.20) // 加速度惩罚权重
    , score_aspect_ratio_weight_(0.10) // 宽高比惩罚权重
    , score_direction_weight_(0.0) // 方向一致性已作为硬性否决，不再评分
{
    LOG_INFO_FMT("PersonBoxRisingDetector initialized with video_path: {}", video_path_);
}

double PersonBoxRisingDetector::calculate_y_center(const core::InferenceResultPacket::BBox& bbox) const {
    return bbox.y + bbox.h / 2.0;
}

double PersonBoxRisingDetector::calculate_x_center(const core::InferenceResultPacket::BBox& bbox) const {
    return bbox.x + bbox.w / 2.0;
}

double PersonBoxRisingDetector::calculate_slope(const std::deque<double>& data) const {
    int n = data.size();
    if (n < 2) return 0.0;
    
    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (int i = 0; i < n; i++) {
        sum_x += i;
        sum_y += data[i];
        sum_xy += i * data[i];
        sum_x2 += i * i;
    }
    
    double denominator = n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-10) return 0.0;
    
    return (n * sum_xy - sum_x * sum_y) / denominator;
}

bool PersonBoxRisingDetector::check_x_displacement(const std::deque<double>& x_history) const {
    if (x_history.size() < 2) return true;
    
    double x_first = x_history.front();
    double x_last = x_history.back();
    double x_displacement = std::abs(x_last - x_first);
    
    return x_displacement < max_x_displacement_;
}

bool PersonBoxRisingDetector::check_height_stability(const std::deque<double>& h_history) const {
    if (h_history.size() < 2) return true;
    
    double h_first = h_history.front();
    double h_last = h_history.back();
    
    if (h_first < 1.0) return true;
    
    double height_ratio = h_last / h_first;
    
    return height_ratio >= min_height_ratio_;
}

bool PersonBoxRisingDetector::check_oscillation(const std::deque<double>& y_history) const {
    if (y_history.size() < 3) return false;
    
    double max_y = *std::max_element(y_history.begin(), y_history.end());
    double min_y = *std::min_element(y_history.begin(), y_history.end());
    double range = max_y - min_y;
    
    int direction_changes = 0;
    for (size_t i = 2; i < y_history.size(); i++) {
        double diff1 = y_history[i-1] - y_history[i-2];
        double diff2 = y_history[i] - y_history[i-1];
        if ((diff1 > 0 && diff2 < 0) || (diff1 < 0 && diff2 > 0)) {
            direction_changes++;
        }
    }
    
    bool has_oscillation = (range > max_oscillation_range_) && (direction_changes > static_cast<int>(y_history.size()) / 3);
    
    return has_oscillation;
}

bool PersonBoxRisingDetector::check_sustain_ratio(const RisingTrackState& state) const {
    int history_size = state.y_history.size();
    if (history_size < required_rising_frames_) return false;
    
    double ratio = static_cast<double>(state.rising_frames) / history_size;
    
    return ratio >= min_sustain_ratio_;
}

bool PersonBoxRisingDetector::check_sustain_ratio_recent(const RisingTrackState& state) const {
    int history_size = static_cast<int>(state.y_history.size());
    int check_window = std::min(window_size_, history_size);
    if (check_window < required_rising_frames_) return false;
    if (state.y_history.size() < static_cast<size_t>(check_window)) return false;
    int rising_in_window = 0;
    for (size_t i = state.y_history.size() - check_window; i < state.y_history.size(); i++) {
        if (i > 0) {
            double diff = state.y_history[i] - state.y_history[i - 1];
            if (diff < 0) rising_in_window++;
        }
    }
    double recent_ratio = static_cast<double>(rising_in_window) / (check_window - 1);
    return recent_ratio >= min_sustain_ratio_;
}

bool PersonBoxRisingDetector::check_acceleration(const std::deque<double>& acceleration_history) const {
    if (acceleration_history.empty()) return true;
    
    for (double acc : acceleration_history) {
        if (std::abs(acc) > max_acceleration_) {
            return false;
        }
    }
    
    return true;
}

bool PersonBoxRisingDetector::check_aspect_ratio_stability(const std::deque<double>& w_history, const std::deque<double>& h_history) const {
    if (w_history.size() < 2 || h_history.size() < 2) return true;
    
    double w_first = w_history.front();
    double h_first = h_history.front();
    double w_last = w_history.back();
    double h_last = h_history.back();
    
    if (w_first < 1.0 || h_first < 1.0) return true;
    
    double ratio_first = w_first / h_first;
    double ratio_last = w_last / h_last;
    
    double ratio_change = std::abs(ratio_last - ratio_first) / ratio_first;
    
    return ratio_change < aspect_ratio_change_threshold_;
}

bool PersonBoxRisingDetector::check_direction_consistency(const std::deque<double>& y_history) const {
    if (y_history.size() < 3) return true;
    
    int up_count = 0;
    int total = 0;
    
    for (size_t i = 1; i < y_history.size(); i++) {
        double diff = y_history[i] - y_history[i-1];
        if (diff < 0) {
            up_count++;
        }
        total++;
    }
    
    double consistency = static_cast<double>(up_count) / total;
    
    return consistency >= min_direction_consistency_;
}

bool PersonBoxRisingDetector::check_camera_x_consistency(const std::vector<int>& rising_track_ids) const {
    if (rising_track_ids.size() < static_cast<size_t>(min_camera_shake_tracks_)) return false;
    double x_sum = 0.0;
    int valid_count = 0;
    for (int track_id : rising_track_ids) {
        auto it = track_rising_state_.find(track_id);
        if (it == track_rising_state_.end()) continue;
        const auto& x_history = it->second.x_history;
        if (x_history.size() < 2) continue;
        double x_first = x_history.front();
        double x_last = x_history.back();
        double x_disp = x_last - x_first;
        if (x_disp > 0) x_sum += 1.0;
        else if (x_disp < 0) x_sum -= 1.0;
        else x_sum += 0.0;
        valid_count++;
    }
    if (valid_count < min_camera_shake_tracks_) return false;
    return std::abs(x_sum) >= static_cast<double>(valid_count);
}

std::deque<double> PersonBoxRisingDetector::smooth_trajectory(const std::deque<double>& y_history) const {
    if (y_history.size() < 2) return y_history;
    
    std::deque<double> smoothed;
    smoothed.push_back(y_history.front());
    
    for (size_t i = 1; i < y_history.size(); i++) {
        double smoothed_value = smooth_factor_ * y_history[i] + (1.0 - smooth_factor_) * smoothed.back();
        smoothed.push_back(smoothed_value);
    }
    
    return smoothed;
}

double PersonBoxRisingDetector::calculate_rising_score(double trend_slope,
                                                        bool x_displacement_ok,
                                                        bool height_stable,
                                                        bool no_oscillation,
                                                        bool acceleration_ok,
                                                        bool aspect_ratio_ok,
                                                        bool direction_ok) const {
    if (trend_slope >= rising_speed_threshold_) {
        return 0.0;
    }
    
    if (!no_oscillation) return 0.0;
    if (!x_displacement_ok) return 0.0;
    if (!direction_ok)   return 0.0;
    
    double score = 1.0;

    if (!height_stable)         score -= score_height_weight_;
    if (!acceleration_ok)       score -= score_acceleration_weight_;
    if (!aspect_ratio_ok)       score -= score_aspect_ratio_weight_;
    
    double slope_strength = std::abs(trend_slope - rising_speed_threshold_) / std::abs(rising_speed_threshold_);
    score += std::min(slope_strength, 1.0) * score_slope_weight_;
    
    return score;
}

RisingResult PersonBoxRisingDetector::process(const std::vector<core::InferenceResultPacket::BBox>& detections) {
    RisingResult result;
    result.is_rising = false;
    
    std::unordered_set<int> active_ids;
    for (const auto& det : detections) {
        if (det.track_id >= 0) {
            active_ids.insert(det.track_id);
            result.active_track_ids.push_back(det.track_id);
        }
    }
    
    std::vector<int> potential_rising_tracks;
    for (const auto& det : detections) {
        if (det.track_id < 0) continue;
        
        bool is_rising = update_track(det.track_id, det);
        auto it = track_rising_state_.find(det.track_id);
        double score = (it != track_rising_state_.end()) ? it->second.rising_score : 0.0;
        
        result.track_rising_states[det.track_id] = is_rising;
        result.track_rising_scores[det.track_id] = score;
        
        if (is_rising) {
            potential_rising_tracks.push_back(det.track_id);
        }
    }
    
    bool is_camera_shake = (static_cast<int>(potential_rising_tracks.size()) >= min_camera_shake_tracks_)
                          && check_camera_x_consistency(potential_rising_tracks);
    
    if (is_camera_shake) {
        for (int track_id : potential_rising_tracks) {
            result.track_rising_states[track_id] = false;
        }
        result.is_rising = false;
        
        std::ostringstream oss;
        oss << "video_path:" << video_path_ 
            << ": 检测到镜头晃动 (同时上升人数=" << potential_rising_tracks.size() << ")";
        LOG_INFO_FMT("{}",oss.str());
    } else {
        for (int track_id : potential_rising_tracks) {
            result.is_rising = true;
            break;
        }
    }
    
    cleanup_old_tracks(active_ids);
    
    return result;
}

bool PersonBoxRisingDetector::update_track(int track_id, const core::InferenceResultPacket::BBox& bbox) {
    if (track_id < 0) {
        return false;
    }
    
    double y_center = calculate_y_center(bbox);
    
    auto it = track_rising_state_.find(track_id);
    if (it == track_rising_state_.end()) {
        track_rising_state_[track_id] = RisingTrackState();
        it = track_rising_state_.find(track_id);
    }
    
    RisingTrackState& state = it->second;
    
    double x_center = calculate_x_center(bbox);
    
    state.y_history.push_back(y_center);
    state.x_history.push_back(x_center);
    state.h_history.push_back(bbox.h);
    state.w_history.push_back(bbox.w);
    
    while (static_cast<int>(state.y_history.size()) > history_length_) {
        state.y_history.pop_front();
        state.x_history.pop_front();
        state.h_history.pop_front();
        state.w_history.pop_front();
    }
    
    while (static_cast<int>(state.speed_history.size()) > history_length_) {
        state.speed_history.pop_front();
    }
    while (static_cast<int>(state.acceleration_history.size()) > history_length_) {
        state.acceleration_history.pop_front();
    }
    
    if (state.y_history.size() >= 2) {
        double current_speed = state.y_history.back() - state.y_history[state.y_history.size() - 2];
        state.speed_history.push_back(current_speed);
        
        if (state.speed_history.size() >= 2) {
            double current_acceleration = state.speed_history.back() - state.speed_history[state.speed_history.size() - 2];
            state.acceleration_history.push_back(current_acceleration);
        }
    }
    
    state.frame_count++;
    
    state.smoothed_y_history = smooth_trajectory(state.y_history);
    
    double trend_slope = 0.0;
    bool x_displacement_ok = check_x_displacement(state.x_history);
    bool height_stable = check_height_stability(state.h_history);
    bool no_oscillation = !check_oscillation(state.smoothed_y_history);
    bool acceleration_ok = check_acceleration(state.acceleration_history);
    bool aspect_ratio_ok = check_aspect_ratio_stability(state.w_history, state.h_history);
    bool direction_ok = check_direction_consistency(state.smoothed_y_history);
    
    if (static_cast<int>(state.smoothed_y_history.size()) >= window_size_) {
        trend_slope = calculate_slope(state.smoothed_y_history);
    }
    
    double rising_score = calculate_rising_score(trend_slope, x_displacement_ok, height_stable, no_oscillation, acceleration_ok, aspect_ratio_ok, direction_ok);
    LOG_INFO_FMT("Track:{}, Rising Score: {}", track_id, rising_score);
    state.rising_score = rising_score;
    
    bool is_currently_rising = (rising_score >= min_rising_score_);
    
    if (is_currently_rising) {
        state.rising_frames++;
        state.non_rising_frames = 0;
    } else {
        state.non_rising_frames++;
        if (state.non_rising_frames > rising_falling_tolerance_) {
            state.rising_frames = std::max(0, state.rising_frames - state.non_rising_frames);
        }
    }
    
    if (state.rising_frames >= required_rising_frames_ && check_sustain_ratio_recent(state)) {
        if (!state.is_rising) {
            std::ostringstream oss;
            oss << "video_path:" << video_path_ 
                << ",Track " << track_id 
                << ": 进入上升状态 (连续上升 " << state.rising_frames 
                << " 帧, 趋势斜率=" << trend_slope << ")";
            LOG_INFO_FMT("{}",oss.str());
        }
        state.is_rising = true;
    } else if (state.non_rising_frames >= required_rising_frames_ || !check_sustain_ratio_recent(state)) {
        if (state.is_rising) {
            std::ostringstream oss;
            oss << "video_path:" << video_path_ 
                << ",Track " << track_id 
                << ": 退出上升状态 (连续非上升 " << state.non_rising_frames 
                << " 帧, 趋势斜率=" << trend_slope << ")";
            LOG_INFO_FMT("{}",oss.str());
        }
        state.is_rising = false;
    }
    
    std::ostringstream oss;
    oss << "video_path:" << video_path_
        << ",Track " << track_id 
        << ": trend_slope=" << trend_slope 
        << ", x_ok=" << (x_displacement_ok ? "1" : "0")
        << ", h_ok=" << (height_stable ? "1" : "0")
        << ", no_osc=" << (no_oscillation ? "1" : "0")
        << ", acc_ok=" << (acceleration_ok ? "1" : "0")
        << ", ar_ok=" << (aspect_ratio_ok ? "1" : "0")
        << ", dir_ok=" << (direction_ok ? "1" : "0")
        << ", score=" << rising_score
        << ", veto=" << (trend_slope >= rising_speed_threshold_ ? "slope" : (!no_oscillation ? "osc" : (!direction_ok ? "dir" : "none")))
        << ", is_rising=" << (state.is_rising ? "true" : "false")
        << ", rising_frames=" << state.rising_frames;
    LOG_INFO_FMT("{}",oss.str());
    
    return state.is_rising;
}

bool PersonBoxRisingDetector::get_rising_state(int track_id) const {
    auto it = track_rising_state_.find(track_id);
    if (it != track_rising_state_.end()) {
        return it->second.is_rising;
    }
    return false;
}

void PersonBoxRisingDetector::cleanup_old_tracks(const std::unordered_set<int>& active_track_ids, int max_age) {
    std::vector<int> inactive_ids;
    for (const auto& pair : track_rising_state_) {
        int track_id = pair.first;
        if (active_track_ids.find(track_id) == active_track_ids.end()) {
            inactive_ids.push_back(track_id);
        }
    }
    
    for (int track_id : inactive_ids) {
        std::ostringstream oss;
        oss << "清理不活跃的跟踪记录: track_id=" << track_id;
        LOG_INFO_FMT("{}",oss.str());
        track_rising_state_.erase(track_id);
    }
}
