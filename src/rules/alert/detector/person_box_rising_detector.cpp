#include "person_box_rising_detector.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <cmath>
#include <algorithm>
#include <sstream>
#include <iostream>

PersonBoxRisingDetector::PersonBoxRisingDetector(const std::string& video_path)
    : video_path_(video_path)
    , video_width_(1920)
    , video_height_(1080)
    , resolution_set_(false)
    // ========== 归一化阈值（与分辨率无关）==========
    , rising_speed_threshold_(-0.003)        // 每帧向上移动超过1.5%画面高度
    , window_size_(10)
    , history_length_(30)
    , required_rising_frames_(5)
    , required_non_rising_frames_(5)
    , min_tracking_frames_(3)
    // 空间稳定性（归一化值）
    , max_x_displacement_(0.015)             // X方向位移不超过1.5%画面宽度
    , min_height_ratio_(0.85)                // 高度不低于初始的85%
    , max_oscillation_range_(0.03)           // 振荡范围不超过3%画面高度
    // 动态阈值
    , max_acceleration_(0.05)                // 归一化加速度上限
    , aspect_ratio_change_threshold_(0.25)   // 宽高比变化不超过25%
    , smooth_factor_(0.6)
    , min_direction_consistency_(0.6)
    // 分数阈值
    , min_rising_score_(0.7)
    , score_smoothing_factor_(0.5)
    // 分数权重（总和为1.0）
    , score_slope_weight_(0.25)
    , score_x_weight_(0.15)
    , score_height_weight_(0.10)
    , score_oscillation_weight_(0.10)
    , score_acceleration_weight_(0.15)
    , score_aspect_ratio_weight_(0.10)
    , score_direction_weight_(0.05)
    // 镜头晃动
    , min_camera_shake_tracks_(2)
{
    LOG_INFO_FMT("PersonBoxRisingDetector initialized with video_path: {}", video_path_);
    LOG_INFO_FMT("Normalized mode: thresholds are resolution-independent");
}

// ========== 归一化辅助函数 ==========

double PersonBoxRisingDetector::normalize_x(double x_pixel) const {
    if (video_width_ <= 0) return 0.0;
    return x_pixel / static_cast<double>(video_width_);
}

double PersonBoxRisingDetector::normalize_y(double y_pixel) const {
    if (video_height_ <= 0) return 0.0;
    return y_pixel / static_cast<double>(video_height_);
}

double PersonBoxRisingDetector::normalize_w(double w_pixel) const {
    if (video_width_ <= 0) return 0.0;
    return w_pixel / static_cast<double>(video_width_);
}

double PersonBoxRisingDetector::normalize_h(double h_pixel) const {
    if (video_height_ <= 0) return 0.0;
    return h_pixel / static_cast<double>(video_height_);
}

double PersonBoxRisingDetector::denormalize_y(double y_norm) const {
    return y_norm * static_cast<double>(video_height_);
}

double PersonBoxRisingDetector::denormalize_h(double h_norm) const {
    return h_norm * static_cast<double>(video_height_);
}

// ========== 分辨率设置 ==========

void PersonBoxRisingDetector::set_video_resolution(int width, int height) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (width > 0 && height > 0) {
        video_width_ = width;
        video_height_ = height;
        resolution_set_ = true;
        LOG_INFO_FMT("Video resolution set to: {}x{}", width, height);
    }
}

std::pair<int, int> PersonBoxRisingDetector::get_video_resolution() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {video_width_, video_height_};
}

// ========== 核心计算函数 ==========

double PersonBoxRisingDetector::calculate_y_center_pixel(const core::InferenceResultPacket::BBox& bbox) const {
    return bbox.y + bbox.h / 2.0;
}

double PersonBoxRisingDetector::calculate_x_center_pixel(const core::InferenceResultPacket::BBox& bbox) const {
    return bbox.x + bbox.w / 2.0;
}

double PersonBoxRisingDetector::calculate_normalized_speed(double y_pixel_current, double y_pixel_prev) const {
    // 归一化速度 = (当前Y - 上一帧Y) / 画面高度
    // 注意：图像坐标系中Y向下增加，所以Y减小=向上移动=负值
    if (video_height_ <= 0) return 0.0;
    return (y_pixel_current - y_pixel_prev) / static_cast<double>(video_height_);
}

double PersonBoxRisingDetector::calculate_slope(const std::deque<double>& data) const {
    int n = static_cast<int>(data.size());
    if (n < 2) return 0.0;

    // 只使用最近 window_size_ 个数据点
    int use_n = std::min(n, window_size_);
    int start_idx = n - use_n;

    double sum_x = 0.0, sum_y = 0.0, sum_xy = 0.0, sum_x2 = 0.0;
    for (int i = 0; i < use_n; i++) {
        double x = static_cast<double>(i);
        double y = data[start_idx + i];
        sum_x += x;
        sum_y += y;
        sum_xy += x * y;
        sum_x2 += x * x;
    }

    double denominator = use_n * sum_x2 - sum_x * sum_x;
    if (std::abs(denominator) < 1e-10) return 0.0;

    return (use_n * sum_xy - sum_x * sum_y) / denominator;
}

// ========== 检查函数（全部基于归一化坐标） ==========

bool PersonBoxRisingDetector::check_x_displacement(const std::deque<double>& x_history_norm) const {
    if (x_history_norm.size() < 2) return true;

    double x_first = x_history_norm.front();
    double x_last = x_history_norm.back();
    double x_displacement = std::abs(x_last - x_first);

    return x_displacement < max_x_displacement_;
}

bool PersonBoxRisingDetector::check_height_stability(const std::deque<double>& h_history_norm) const {
    if (h_history_norm.size() < 2) return true;

    double h_first = h_history_norm.front();
    double h_last = h_history_norm.back();

    // 归一化高度接近0表示无效
    if (h_first < 1e-6) return true;

    double height_ratio = h_last / h_first;

    return height_ratio >= min_height_ratio_;
}

bool PersonBoxRisingDetector::check_oscillation(const std::deque<double>& y_history_norm) const {
    if (y_history_norm.size() < 3) return false;

    double max_y = *std::max_element(y_history_norm.begin(), y_history_norm.end());
    double min_y = *std::min_element(y_history_norm.begin(), y_history_norm.end());
    double range = max_y - min_y;

    int direction_changes = 0;
    for (size_t i = 2; i < y_history_norm.size(); i++) {
        double diff1 = y_history_norm[i-1] - y_history_norm[i-2];
        double diff2 = y_history_norm[i] - y_history_norm[i-1];
        if ((diff1 > 0 && diff2 < 0) || (diff1 < 0 && diff2 > 0)) {
            direction_changes++;
        }
    }

    bool has_oscillation = (range > max_oscillation_range_) 
                        && (direction_changes > static_cast<int>(y_history_norm.size()) / 3);

    return has_oscillation;
}

bool PersonBoxRisingDetector::check_sustain_ratio_recent(const RisingTrackState& state) const {
    int history_size = static_cast<int>(state.y_history_norm.size());
    int check_window = std::min(window_size_, history_size);
    if (check_window < required_rising_frames_) return false;
    if (state.y_history_norm.size() < static_cast<size_t>(check_window)) return false;

    int rising_in_window = 0;
    for (size_t i = state.y_history_norm.size() - check_window; i < state.y_history_norm.size(); i++) {
        if (i > 0) {
            double diff = state.y_history_norm[i] - state.y_history_norm[i - 1];
            if (diff < 0) rising_in_window++;
        }
    }
    double recent_ratio = static_cast<double>(rising_in_window) / (check_window - 1);
    return recent_ratio >= min_direction_consistency_;
}

bool PersonBoxRisingDetector::check_acceleration(const std::deque<double>& acceleration_history_norm) const {
    if (acceleration_history_norm.empty()) return true;

    for (double acc : acceleration_history_norm) {
        if (std::abs(acc) > max_acceleration_) {
            return false;
        }
    }

    return true;
}

bool PersonBoxRisingDetector::check_aspect_ratio_stability(
    const std::deque<double>& w_history_norm, 
    const std::deque<double>& h_history_norm) const {

    if (w_history_norm.size() < 2 || h_history_norm.size() < 2) return true;

    double w_first = w_history_norm.front();
    double h_first = h_history_norm.front();
    double w_last = w_history_norm.back();
    double h_last = h_history_norm.back();

    if (h_first < 1e-6 || w_first < 1e-6) return true;

    double ratio_first = w_first / h_first;
    double ratio_last = w_last / h_last;

    double ratio_change = std::abs(ratio_last - ratio_first) / ratio_first;

    return ratio_change < aspect_ratio_change_threshold_;
}

bool PersonBoxRisingDetector::check_direction_consistency(const std::deque<double>& y_history_norm) const {
    if (y_history_norm.size() < 3) return true;

    int up_count = 0;
    int total = 0;

    for (size_t i = 1; i < y_history_norm.size(); i++) {
        double diff = y_history_norm[i] - y_history_norm[i-1];
        if (diff < 0) {
            up_count++;
        }
        total++;
    }

    if (total == 0) return true;
    double consistency = static_cast<double>(up_count) / total;

    return consistency >= min_direction_consistency_;
}

// ========== 镜头晃动检测（基于归一化坐标） ==========

bool PersonBoxRisingDetector::check_camera_x_consistency(const std::vector<int>& rising_track_ids) const {
    if (rising_track_ids.size() < static_cast<size_t>(min_camera_shake_tracks_)) return false;

    double x_sum = 0.0;
    int valid_count = 0;

    for (int track_id : rising_track_ids) {
        auto it = track_rising_state_.find(track_id);
        if (it == track_rising_state_.end()) continue;
        const auto& x_history = it->second.x_history_norm;
        if (x_history.size() < 2) continue;

        double x_first = x_history.front();
        double x_last = x_history.back();
        double x_disp = x_last - x_first;

        if (x_disp > 0) x_sum += 1.0;
        else if (x_disp < 0) x_sum -= 1.0;

        valid_count++;
    }

    if (valid_count < min_camera_shake_tracks_) return false;
    return std::abs(x_sum) >= static_cast<double>(valid_count);
}

bool PersonBoxRisingDetector::check_camera_y_consistency(const std::vector<int>& rising_track_ids) const {
    if (rising_track_ids.size() < static_cast<size_t>(min_camera_shake_tracks_)) return false;

    double y_sum = 0.0;
    int valid_count = 0;

    for (int track_id : rising_track_ids) {
        auto it = track_rising_state_.find(track_id);
        if (it == track_rising_state_.end()) continue;
        const auto& y_history = it->second.y_history_norm;
        if (y_history.size() < 2) continue;

        double y_first = y_history.front();
        double y_last = y_history.back();
        double y_disp = y_last - y_first;  // 负值表示上升

        if (y_disp < 0) y_sum += 1.0;      // 上升
        else if (y_disp > 0) y_sum -= 1.0; // 下降

        valid_count++;
    }

    if (valid_count < min_camera_shake_tracks_) return false;
    return y_sum >= static_cast<double>(valid_count);
}

bool PersonBoxRisingDetector::check_camera_shake(const std::vector<int>& rising_track_ids) const {
    return check_camera_x_consistency(rising_track_ids) 
        && check_camera_y_consistency(rising_track_ids);
}

// ========== 平滑轨迹（归一化） ==========

std::deque<double> PersonBoxRisingDetector::smooth_trajectory(const std::deque<double>& y_history_norm) const {
    if (y_history_norm.size() < 2) return y_history_norm;

    std::deque<double> smoothed;
    smoothed.push_back(y_history_norm.front());

    for (size_t i = 1; i < y_history_norm.size(); i++) {
        double smoothed_value = smooth_factor_ * y_history_norm[i] + (1.0 - smooth_factor_) * smoothed.back();
        smoothed.push_back(smoothed_value);
    }

    return smoothed;
}

// ========== 分数计算（归一化版本） ==========

double PersonBoxRisingDetector::calculate_rising_score(
    double trend_slope_norm,
    bool x_displacement_ok,
    bool height_stable,
    bool no_oscillation,
    bool acceleration_ok,
    bool aspect_ratio_ok,
    bool direction_ok,
    double sustain_ratio) const {

    // 硬门限：趋势斜率必须小于阈值（负值表示上升）
    if (trend_slope_norm >= rising_speed_threshold_) {
        return 0.0;
    }

    // 硬门限：方向一致性必须满足
    if (!direction_ok) return 0.0;

    // 基础分
    double score = 1.0;

    // 软惩罚（使用配置的权重）
    if (!x_displacement_ok)     score -= score_x_weight_;
    if (!height_stable)         score -= score_height_weight_;
    if (!no_oscillation)        score -= score_oscillation_weight_;
    if (!acceleration_ok)       score -= score_acceleration_weight_;
    if (!aspect_ratio_ok)       score -= score_aspect_ratio_weight_;

    // 斜率强度奖励（归一化）
    double slope_strength = std::abs(trend_slope_norm - rising_speed_threshold_) 
                          / std::abs(rising_speed_threshold_);
    score += std::min(slope_strength, 2.0) * score_slope_weight_;

    // 持续比例奖励
    score += sustain_ratio * score_direction_weight_;

    // 限制到 [0, 1]
    return std::max(0.0, std::min(1.0, score));
}

// ========== 状态机 ==========

ClimbEvent PersonBoxRisingDetector::evaluate_event(
    const RisingTrackState& state, bool is_currently_rising) const {

    if (state.consecutive_tracking < min_tracking_frames_) {
        return ClimbEvent::NO_SIGNAL;
    }

    if (is_currently_rising) {
        if (state.rising_frames >= required_rising_frames_) {
            return ClimbEvent::RISING_CONFIRMED;
        } else {
            return ClimbEvent::RISING_DETECTED;
        }
    } else {
        if (state.non_rising_frames >= required_non_rising_frames_) {
            return ClimbEvent::RISING_LOST;
        } else {
            return ClimbEvent::NO_SIGNAL;
        }
    }
}

ClimbState PersonBoxRisingDetector::get_next_state(ClimbState current, ClimbEvent event) const {
    switch (current) {
        case ClimbState::IDLE:
            switch (event) {
                case ClimbEvent::NO_SIGNAL:        return ClimbState::TRACKING;
                case ClimbEvent::RISING_DETECTED:  return ClimbState::TRACKING;
                case ClimbEvent::RISING_CONFIRMED: return ClimbState::CLIMBING;
                default: return current;
            }

        case ClimbState::TRACKING:
            switch (event) {
                case ClimbEvent::NO_SIGNAL:        return ClimbState::TRACKING;
                case ClimbEvent::RISING_DETECTED:  return ClimbState::TRACKING;
                case ClimbEvent::RISING_CONFIRMED: return ClimbState::CLIMBING;
                case ClimbEvent::RISING_LOST:      return ClimbState::IDLE;
                case ClimbEvent::TRACK_LOST:       return ClimbState::IDLE;
                default: return current;
            }

        case ClimbState::CLIMBING:
            switch (event) {
                case ClimbEvent::NO_SIGNAL:        return ClimbState::CLIMBING;
                case ClimbEvent::RISING_DETECTED:  return ClimbState::CLIMBING;
                case ClimbEvent::RISING_CONFIRMED: return ClimbState::CLIMBING;
                case ClimbEvent::RISING_LOST:      return ClimbState::TRACKING;
                case ClimbEvent::TRACK_LOST:       return ClimbState::IDLE;
                default: return current;
            }

        case ClimbState::EXITED:
            switch (event) {
                case ClimbEvent::TRACK_LOST:       return ClimbState::IDLE;
                default: return ClimbState::TRACKING;
            }

        default:
            return ClimbState::IDLE;
    }
}

void PersonBoxRisingDetector::on_enter_state(RisingTrackState& state, ClimbState new_state, int frame_id) {
    state.state_entry_frame = frame_id;

    switch (new_state) {
        case ClimbState::CLIMBING:
            LOG_INFO_FMT("video_path:{}, Track {}: 进入攀爬状态 (连续上升 {} 帧, 平滑分数={:.3f})",
                        video_path_, state.frame_count, state.rising_frames, state.smoothed_score);
            break;
        case ClimbState::TRACKING:
            LOG_INFO_FMT("video_path:{}, Track {}: 进入跟踪状态", video_path_, state.frame_count);
            break;
        case ClimbState::IDLE:
            break;
        default:
            break;
    }
}

void PersonBoxRisingDetector::on_exit_state(RisingTrackState& state, ClimbState old_state, int frame_id) {
    switch (old_state) {
        case ClimbState::CLIMBING:
            LOG_INFO_FMT("video_path:{}, Track {}: 退出攀爬状态 (连续非上升 {} 帧, 持续 {} 帧)",
                        video_path_, state.frame_count, state.non_rising_frames,
                        frame_id - state.state_entry_frame);
            break;
        default:
            break;
    }
}

void PersonBoxRisingDetector::transition_state(RisingTrackState& state, ClimbEvent event, int frame_id) {
    ClimbState old_state = state.current_state;
    ClimbState new_state = get_next_state(old_state, event);

    if (new_state != old_state) {
        on_exit_state(state, old_state, frame_id);
        state.current_state = new_state;
        on_enter_state(state, new_state, frame_id);
    }
}

// ========== 主更新函数（归一化版本） ==========

bool PersonBoxRisingDetector::update_track(int track_id, const core::InferenceResultPacket::BBox& bbox, int frame_id) {
    if (track_id < 0) {
        return false;
    }

    // 提取像素值
    double y_center_pixel = calculate_y_center_pixel(bbox);
    double x_center_pixel = calculate_x_center_pixel(bbox);
    double w_pixel = bbox.w;
    double h_pixel = bbox.h;

    // 归一化
    double y_center_norm = normalize_y(y_center_pixel);
    double x_center_norm = normalize_x(x_center_pixel);
    double w_norm = normalize_w(w_pixel);
    double h_norm = normalize_h(h_pixel);

    auto it = track_rising_state_.find(track_id);
    if (it == track_rising_state_.end()) {
        track_rising_state_[track_id] = RisingTrackState();
        it = track_rising_state_.find(track_id);
        it->second.video_width = video_width_;
        it->second.video_height = video_height_;
    }

    RisingTrackState& state = it->second;

    // 更新帧号信息
    state.last_update_frame = frame_id;
    state.frame_count++;
    state.consecutive_tracking++;

    // 存储归一化坐标
    state.y_history_norm.push_back(y_center_norm);
    state.x_history_norm.push_back(x_center_norm);
    state.h_history_norm.push_back(h_norm);
    state.w_history_norm.push_back(w_norm);

    // 同时存储像素值用于日志
    state.y_pixel_history.push_back(y_center_pixel);
    state.x_pixel_history.push_back(x_center_pixel);
    state.h_pixel_history.push_back(h_pixel);
    state.w_pixel_history.push_back(w_pixel);

    // 维护历史长度
    while (static_cast<int>(state.y_history_norm.size()) > history_length_) {
        state.y_history_norm.pop_front();
        state.x_history_norm.pop_front();
        state.h_history_norm.pop_front();
        state.w_history_norm.pop_front();
        state.y_pixel_history.pop_front();
        state.x_pixel_history.pop_front();
        state.h_pixel_history.pop_front();
        state.w_pixel_history.pop_front();
    }

    // 计算归一化速度和加速度
    if (state.y_history_norm.size() >= 2) {
        double current_speed = state.y_history_norm.back() - state.y_history_norm[state.y_history_norm.size() - 2];
        state.speed_history_norm.push_back(current_speed);

        if (state.speed_history_norm.size() >= 2) {
            double current_acceleration = state.speed_history_norm.back() - state.speed_history_norm[state.speed_history_norm.size() - 2];
            state.acceleration_history_norm.push_back(current_acceleration);
        }
    }

    // 维护速度和加速度历史长度
    while (static_cast<int>(state.speed_history_norm.size()) > window_size_) {
        state.speed_history_norm.pop_front();
    }
    while (static_cast<int>(state.acceleration_history_norm.size()) > window_size_) {
        state.acceleration_history_norm.pop_front();
    }

    // 平滑Y轨迹（归一化）
    state.smoothed_y_history_norm = smooth_trajectory(state.y_history_norm);

    // ========== 执行各项检查（全部基于归一化坐标） ==========
    double trend_slope = 0.0;
    bool x_displacement_ok = check_x_displacement(state.x_history_norm);
    bool height_stable = check_height_stability(state.h_history_norm);
    bool no_oscillation = !check_oscillation(state.smoothed_y_history_norm);
    bool acceleration_ok = check_acceleration(state.acceleration_history_norm);
    bool aspect_ratio_ok = check_aspect_ratio_stability(state.w_history_norm, state.h_history_norm);
    bool direction_ok = check_direction_consistency(state.smoothed_y_history_norm);

    if (static_cast<int>(state.smoothed_y_history_norm.size()) >= window_size_) {
        trend_slope = calculate_slope(state.smoothed_y_history_norm);
    }

    // 持续比例检查
    double sustain_ratio = check_sustain_ratio_recent(state) ? 1.0 : 0.0;

    // 计算分数（归一化）
    double rising_score = calculate_rising_score(
        trend_slope, x_displacement_ok, height_stable, no_oscillation,
        acceleration_ok, aspect_ratio_ok, direction_ok, sustain_ratio
    );

    // 分数平滑（防止抖动）
    if (state.smoothed_score > 0) {
        state.smoothed_score = score_smoothing_factor_ * rising_score 
                             + (1.0 - score_smoothing_factor_) * state.smoothed_score;
    } else {
        state.smoothed_score = rising_score;
    }
    state.rising_score = state.smoothed_score;

    LOG_INFO_FMT("Track:{}, Rising Score: {:.3f}", track_id, state.rising_score);

    // 使用平滑后的分数判断
    bool is_currently_rising = (state.smoothed_score >= min_rising_score_);

    // 逐帧衰减机制
    if (is_currently_rising) {
        state.rising_frames++;
        state.non_rising_frames = std::max(0, state.non_rising_frames - 1);
    } else {
        state.non_rising_frames++;
        state.rising_frames = std::max(0, state.rising_frames - 1);
    }

    // 状态机转换
    ClimbEvent event = evaluate_event(state, is_currently_rising);
    transition_state(state, event, frame_id);

    // 兼容旧接口
    state.is_rising = (state.current_state == ClimbState::CLIMBING);

    // 详细调试日志（同时输出归一化值和像素值）
    std::ostringstream oss;
    oss << "video_path:" << video_path_
        << ",Track " << track_id 
        << ": trend_slope=" << std::fixed << std::setprecision(6) << trend_slope 
        << ", x_ok=" << (x_displacement_ok ? "1" : "0")
        << ", h_ok=" << (height_stable ? "1" : "0")
        << ", no_osc=" << (no_oscillation ? "1" : "0")
        << ", acc_ok=" << (acceleration_ok ? "1" : "0")
        << ", ar_ok=" << (aspect_ratio_ok ? "1" : "0")
        << ", dir_ok=" << (direction_ok ? "1" : "0")
        << ", sustain=" << sustain_ratio
        << ", score=" << std::setprecision(3) << state.rising_score
        << ", y_norm=" << y_center_norm
        << ", h_norm=" << h_norm
        << ", state=" << static_cast<int>(state.current_state)
        << ", rising_frames=" << state.rising_frames
        << ", non_rising=" << state.non_rising_frames;
    LOG_INFO_FMT("{}", oss.str());

    return state.is_rising;
}

// ========== 批量处理接口 ==========

RisingResult PersonBoxRisingDetector::process(
    const std::vector<core::InferenceResultPacket::BBox>& detections, int frame_id) {

    std::lock_guard<std::mutex> lock(mutex_);

    RisingResult result;
    result.frame_id = frame_id;
    result.is_rising = false;
    result.is_camera_shake = false;

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

        bool is_rising = update_track(det.track_id, det, frame_id);
        auto it = track_rising_state_.find(det.track_id);
        double score = (it != track_rising_state_.end()) ? it->second.rising_score : 0.0;
        ClimbState climb_state = (it != track_rising_state_.end()) ? it->second.current_state : ClimbState::IDLE;

        result.track_rising_states[det.track_id] = is_rising;
        result.track_rising_scores[det.track_id] = score;
        result.track_climb_states[det.track_id] = climb_state;

        if (is_rising) {
            potential_rising_tracks.push_back(det.track_id);
        }
    }

    // 镜头晃动检测（基于归一化坐标）
    bool is_camera_shake = (static_cast<int>(potential_rising_tracks.size()) >= min_camera_shake_tracks_)
                          && check_camera_shake(potential_rising_tracks);

    if (is_camera_shake) {
        result.is_camera_shake = true;
        for (int track_id : potential_rising_tracks) {
            result.track_rising_states[track_id] = false;
            auto it = track_rising_state_.find(track_id);
            if (it != track_rising_state_.end() && it->second.current_state == ClimbState::CLIMBING) {
                transition_state(it->second, ClimbEvent::RISING_LOST, frame_id);
                it->second.is_rising = false;
            }
        }
        result.is_rising = false;

        LOG_INFO_FMT("video_path:{}: 检测到镜头晃动 (同时上升人数={}, X和Y方向均一致)", 
                     video_path_, potential_rising_tracks.size());
    } else {
        for (const auto& pair : result.track_rising_states) {
            if (pair.second) {
                result.is_rising = true;
                break;
            }
        }
    }

    cleanup_old_tracks(active_ids);

    return result;
}

// ========== 查询接口 ==========

bool PersonBoxRisingDetector::get_rising_state(int track_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = track_rising_state_.find(track_id);
    if (it != track_rising_state_.end()) {
        return it->second.is_rising;
    }
    return false;
}

ClimbState PersonBoxRisingDetector::get_climb_state(int track_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = track_rising_state_.find(track_id);
    if (it != track_rising_state_.end()) {
        return it->second.current_state;
    }
    return ClimbState::IDLE;
}

std::string PersonBoxRisingDetector::get_climb_state_string(int track_id) const {
    ClimbState state = get_climb_state(track_id);
    switch (state) {
        case ClimbState::IDLE:     return "IDLE";
        case ClimbState::TRACKING: return "TRACKING";
        case ClimbState::CLIMBING: return "CLIMBING";
        case ClimbState::EXITED:   return "EXITED";
        default: return "UNKNOWN";
    }
}

// ========== 清理旧Track（使用max_age） ==========

void PersonBoxRisingDetector::cleanup_old_tracks(const std::unordered_set<int>& active_track_ids, int max_age) {
    if (max_age <= 0) max_age = 50;

    std::vector<int> to_remove;
    int current_frame = -1;

    for (const auto& pair : track_rising_state_) {
        current_frame = std::max(current_frame, pair.second.last_update_frame);
    }

    if (current_frame < 0) return;

    for (const auto& pair : track_rising_state_) {
        int track_id = pair.first;
        const RisingTrackState& state = pair.second;

        if (active_track_ids.find(track_id) == active_track_ids.end()) {
            int age = current_frame - state.last_update_frame;
            if (age >= max_age) {
                to_remove.push_back(track_id);
            }
        }
    }

    for (int track_id : to_remove) {
        auto it = track_rising_state_.find(track_id);
        if (it != track_rising_state_.end()) {
            if (it->second.current_state == ClimbState::CLIMBING) {
                LOG_INFO_FMT("video_path:{}, Track {}: 因超时被清理 (最后更新帧={}, age={})",
                            video_path_, track_id, it->second.last_update_frame,
                            current_frame - it->second.last_update_frame);
            }
            track_rising_state_.erase(it);
        }
    }
}