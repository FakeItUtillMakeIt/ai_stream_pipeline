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
    
    bool has_oscillation = (range > max_oscillation_range_) && (direction_changes > y_history.size() / 3);
    
    return has_oscillation;
}

bool PersonBoxRisingDetector::check_sustain_ratio(const RisingTrackState& state) const {
    int history_size = state.y_history.size();
    if (history_size < required_rising_frames_) return false;
    
    double ratio = static_cast<double>(state.rising_frames) / history_size;
    
    return ratio >= min_sustain_ratio_;
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
        result.track_rising_states[det.track_id] = is_rising;
        
        if (is_rising) {
            potential_rising_tracks.push_back(det.track_id);
        }
    }
    
    bool is_camera_shake = (static_cast<int>(potential_rising_tracks.size()) >= min_camera_shake_tracks_);
    
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
    while (static_cast<int>(state.y_history.size()) > history_length_) {
        state.y_history.pop_front();
        state.x_history.pop_front();
        state.h_history.pop_front();
    }
    
    state.frame_count++;
    
    double trend_slope = 0.0;
    bool x_displacement_ok = check_x_displacement(state.x_history);
    bool height_stable = check_height_stability(state.h_history);
    bool no_oscillation = !check_oscillation(state.y_history);
    
    if (static_cast<int>(state.y_history.size()) >= window_size_) {
        trend_slope = calculate_slope(state.y_history);
    }
    
    bool is_currently_rising = (trend_slope < rising_speed_threshold_) && x_displacement_ok && height_stable && no_oscillation;
    
    if (is_currently_rising) {
        state.rising_frames++;
        state.non_rising_frames = 0;
    } else {
        state.rising_frames = 0;
        state.non_rising_frames++;
    }
    
    if (state.rising_frames >= required_rising_frames_) {
        if (!state.is_rising) {
            std::ostringstream oss;
            oss << "video_path:" << video_path_ 
                << ",Track " << track_id 
                << ": 进入上升状态 (连续上升 " << state.rising_frames 
                << " 帧, 趋势斜率=" << trend_slope << ")";
            LOG_INFO_FMT("{}",oss.str());
        }
        state.is_rising = true;
    } else if (state.non_rising_frames >= required_rising_frames_) {
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
        << ", x_ok=" << (x_displacement_ok ? "true" : "false")
        << ", h_ok=" << (height_stable ? "true" : "false")
        << ", no_osc=" << (no_oscillation ? "true" : "false")
        << ", is_rising=" << (state.is_rising ? "true" : "false")
        << ", rising_frames=" << state.rising_frames
        << ", frame_count=" << state.frame_count;
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
