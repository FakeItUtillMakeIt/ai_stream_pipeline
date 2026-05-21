#include "moving_phonecall_detector.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>

MovingPhonecallDetector::MovingPhonecallDetector(const std::string& video_path)
    : video_path_(video_path)
    , moving_speed_threshold_(10.0)
    , static_speed_threshold_(5.0)
    , window_size_(5)
    , history_length_(30)
    , required_moving_frames_(2) {
    
    LOG_INFO_FMT("MovingPhonecallDetector initialized with video_path: {}", video_path_);
}

std::pair<double, double> MovingPhonecallDetector::calculate_center(const std::vector<double>& person_box) const {
    if (person_box.size() < 4) {
        return {0.0, 0.0};
    }
    double center_x = (person_box[0] + person_box[2]) / 2.0;
    double center_y = (person_box[1] + person_box[3]) / 2.0;
    return {center_x, center_y};
}

double MovingPhonecallDetector::calculate_distance(PixelPoint p1, PixelPoint p2) const {
    double dx = p2.x - p1.x;
    double dy = p2.y - p1.y;
    return std::sqrt(dx * dx + dy * dy);
}

bool MovingPhonecallDetector::update_track(int track_id, const std::vector<double>& person_box, int frame_id) {
    if (track_id < 0) {
        return false;
    }
    
    // 计算人体中心点
    auto [center_x, center_y] = calculate_center(person_box);
    
    // 初始化跟踪记录
    auto it = track_moving_state_.find(track_id);
    if (it == track_moving_state_.end()) {
        track_moving_state_[track_id] = TrackState();
        it = track_moving_state_.find(track_id);
    }
    
    TrackState& state = it->second;
    
    // 添加当前位置到历史
    state.history.emplace_back(PixelPoint(center_x, center_y), frame_id);
    while (static_cast<int>(state.history.size()) > history_length_) {
        state.history.pop_front();
    }
    
    // 计算当前帧速度（与前一帧比较）
    double current_speed = 0.0;
    if (state.history.size() >= 2) {
        const auto& prev = state.history[state.history.size() - 2];
        const auto& curr = state.history[state.history.size() - 1];
        int frame_diff = std::get<1>(curr) - std::get<1>(prev);
        if (frame_diff > 0) {
            double displacement = calculate_distance(
                std::get<0>(prev),
                std::get<0>(curr)
            );
            current_speed = displacement / frame_diff;
        }
    }
    
    // 添加到速度历史
    state.speeds.push_back(current_speed);
    while (static_cast<int>(state.speeds.size()) > window_size_) {
        state.speeds.pop_front();
    }
    
    // 计算窗口内的平均速度
    double avg_speed = current_speed;
    if (static_cast<int>(state.speeds.size()) >= window_size_ / 2) {
        double sum = 0.0;
        for (double speed : state.speeds) {
            sum += speed;
        }
        avg_speed = sum / state.speeds.size();
    }
    
    // 判断当前是否移动
    bool is_currently_moving = avg_speed > moving_speed_threshold_;
    
    // 更新连续计数
    if (is_currently_moving) {
        state.moving_frames++;
        state.static_frames = 0;
    } else {
        state.moving_frames = 0;
        state.static_frames++;
    }
    
    // 根据连续帧数判断状态
    if (state.moving_frames >= required_moving_frames_) {
        if (!state.is_moving) {
            std::ostringstream oss;
            oss << "video_path:" << video_path_ 
                << ",Track " << track_id 
                << ": 进入移动状态 (连续移动 " << state.moving_frames 
                << " 帧, 平均速度=" << avg_speed << ")";
            LOG_INFO_FMT("{}",oss.str());
        }
        state.is_moving = true;
    } else if (state.static_frames >= required_moving_frames_) {
        if (state.is_moving) {
            std::ostringstream oss;
            oss << "video_path:" << video_path_ 
                << ",Track " << track_id 
                << ": 退出移动状态 (连续静止 " << state.static_frames 
                << " 帧, 平均速度=" << avg_speed << ")";
            LOG_INFO_FMT("{}",oss.str());
        }
        state.is_moving = false;
    }
    
    // 详细日志
    std::ostringstream oss;
    oss << "video_path:" << video_path_
        << ",Track " << track_id 
        << ": speed=" << current_speed 
        << ", avg_speed=" << avg_speed 
        << ", is_moving=" << (state.is_moving ? "true" : "false")
        << ", moving_frames=" << state.moving_frames;
    LOG_INFO_FMT("{}",oss.str());
    
    return state.is_moving;
}

bool MovingPhonecallDetector::get_moving_state(int track_id) const {
    auto it = track_moving_state_.find(track_id);
    if (it != track_moving_state_.end()) {
        return it->second.is_moving;
    }
    return false;
}

void MovingPhonecallDetector::cleanup_old_tracks(const std::unordered_set<int>& active_track_ids, int max_age) {
    std::vector<int> inactive_ids;
    for (const auto& pair : track_moving_state_) {
        int track_id = pair.first;
        if (active_track_ids.find(track_id) == active_track_ids.end()) {
            inactive_ids.push_back(track_id);
        }
    }
    
    for (int track_id : inactive_ids) {
        std::ostringstream oss;
        oss << "清理不活跃的跟踪记录: track_id=" << track_id;
        LOG_INFO_FMT("{}",oss.str());
        track_moving_state_.erase(track_id);
    }
}