#include "climbing_detector.h"
#include <algorithm>
#include <cmath>

ClimbingDetector::ClimbingDetector()
    : vertical_speed_threshold_(8.0f), ascent_speed_threshold_(6.0f), window_size_(5), history_length_(30), required_climbing_frames_(5), height_change_ratio_(0.3f)
{
}

bool ClimbingDetector::updateDetection(int track_id, const ai_stream::core::InferenceResultPacket::BBox &person_box, int frame_id)
{
    if (track_id < 0)
    {
        return false;
    }

    // 计算人体中心点和高度
    float center_x = person_box.x + person_box.w / 2.0f;
    float center_y = person_box.y + person_box.h / 2.0f;
    float person_height = person_box.h;
    float person_y_bottom = person_box.y + person_box.h;

    // 初始化跟踪记录
    if (track_climbing_state_.find(track_id) == track_climbing_state_.end())
    {
        track_climbing_state_[track_id] = TrackState();
    }

    TrackState &state = track_climbing_state_[track_id];

    // 添加当前位置到历史
    state.history.emplace_back(center_x, center_y, person_y_bottom, person_height, frame_id);
    if (state.history.size() > static_cast<size_t>(history_length_))
    {
        state.history.pop_front();
    }

    // 计算当前帧垂直速度（与前一帧比较）
    float current_vertical_speed = 0.0f;
    float current_height_change = 0.0f;
    bool height_upward_movement = false;

    if (state.history.size() >= 2)
    {
        const auto &prev = state.history[state.history.size() - 2];
        const auto &curr = state.history[state.history.size() - 1];
        int frame_diff = curr.frame_id - prev.frame_id;

        if (frame_diff > 0)
        {
            // 计算垂直方向位移（Y轴变化，负值表示向上移动）
            float vertical_displacement = curr.bottom_y - prev.bottom_y;
            current_vertical_speed = std::abs(vertical_displacement) / frame_diff;
            height_upward_movement = (vertical_displacement < 0); // 向上移动

            // 计算人体高度变化
            current_height_change = curr.height - prev.height;
        }
    }

    // 添加到垂直速度历史
    state.vertical_speeds.push_back(current_vertical_speed);
    if (state.vertical_speeds.size() > static_cast<size_t>(window_size_))
    {
        state.vertical_speeds.pop_front();
    }

    // 计算窗口内的平均垂直速度
    float avg_vertical_speed = current_vertical_speed;
    if (state.vertical_speeds.size() >= static_cast<size_t>(window_size_ / 2))
    {
        float sum = 0.0f;
        for (float speed : state.vertical_speeds)
        {
            sum += speed;
        }
        avg_vertical_speed = sum / state.vertical_speeds.size();
    }

    // 判断是否正在攀爬
    // 条件1：垂直速度超过阈值
    bool speed_condition = (avg_vertical_speed > vertical_speed_threshold_);

    // 条件2：上升速度条件（针对攀爬场景）
    bool ascent_condition = (avg_vertical_speed > ascent_speed_threshold_ && height_upward_movement);

    // 条件3：人体高度变化（手部伸展等特征）
    bool height_change_condition = (std::abs(current_height_change) > (person_height * height_change_ratio_));

    // 综合判断（满足至少两个条件）
    int conditions_met = (speed_condition ? 1 : 0) +
                         (ascent_condition ? 1 : 0) +
                         (height_change_condition ? 1 : 0);
    bool is_currently_climbing = (conditions_met >= 2);

    // 更新连续计数
    if (is_currently_climbing)
    {
        state.climbing_frames++;
        state.non_climbing_frames = 0;
    }
    else
    {
        state.climbing_frames = 0;
        state.non_climbing_frames++;
    }

    // 根据连续帧数判断状态
    if (state.climbing_frames >= required_climbing_frames_)
    {
        if (!state.is_climbing)
        {
            LOG_INFO_FMT("Track {}: 进入攀爬状态 (连续攀爬 {} 帧, 垂直速度={.2f}, 向上移动={})",
                         track_id, state.climbing_frames, avg_vertical_speed, height_upward_movement);
        }
        state.is_climbing = true;
    }
    else if (state.non_climbing_frames >= required_climbing_frames_)
    {
        if (state.is_climbing)
        {
            LOG_INFO_FMT("Track {}: 攀爬结束 (连续攀爬 {} 帧, 垂直速度={.2f})",
                         track_id, state.climbing_frames, avg_vertical_speed);
        }
        state.is_climbing = false;
    }

    LOG_DEBUG_FMT("Track {}: vertical_speed={.2f}, avg_vertical_speed={.2f}, height_change={.2f}, is_climbing={}, climbing_frames={}",
                  track_id, current_vertical_speed, avg_vertical_speed, current_height_change,state.is_climbing ? 1 : 0, state.climbing_frames);

    return state.is_climbing;
}

bool ClimbingDetector::getClimbingState(int track_id) const
{
    auto it = track_climbing_state_.find(track_id);
    if (it != track_climbing_state_.end())
    {
        return it->second.is_climbing;
    }
    return false;
}

std::unique_ptr<ClimbingDetector::ClimbingDetails>
ClimbingDetector::getClimbingDetails(int track_id) const
{
    auto it = track_climbing_state_.find(track_id);
    if (it == track_climbing_state_.end())
    {
        return nullptr;
    }

    const TrackState &state = it->second;
    auto details = std::make_unique<ClimbingDetails>();
    details->is_climbing = state.is_climbing;
    details->climbing_frames = state.climbing_frames;

    // 计算平均垂直速度
    if (!state.vertical_speeds.empty())
    {
        float sum = 0.0f;
        for (float speed : state.vertical_speeds)
        {
            sum += speed;
        }
        details->avg_vertical_speed = sum / state.vertical_speeds.size();
    }
    else
    {
        details->avg_vertical_speed = 0.0f;
    }

    return details;
}

void ClimbingDetector::cleanupOldTracks(const std::vector<int> &active_track_ids, int max_age)
{
    // max_age 参数未使用，保留接口兼容性

    // 构建活跃track_id的集合以加快查找
    std::vector<int> inactive_ids;

    for (const auto &pair : track_climbing_state_)
    {
        int track_id = pair.first;
        bool is_active = false;

        for (int active_id : active_track_ids)
        {
            if (track_id == active_id)
            {
                is_active = true;
                break;
            }
        }

        if (!is_active)
        {
            inactive_ids.push_back(track_id);
        }
    }

    for (int track_id : inactive_ids)
    {
        LOG_DEBUG_FMT("清理不活跃的攀爬检测记录: track_id={}",track_id);
        track_climbing_state_.erase(track_id);
    }
}

void ClimbingDetector::removeTrack(int track_id)
{
    auto it = track_climbing_state_.find(track_id);
    if (it != track_climbing_state_.end())
    {
        LOG_DEBUG_FMT("手动移除攀爬检测记录: track_id={}",track_id);
        track_climbing_state_.erase(it);
    }
}

void ClimbingDetector::clearAll()
{
    track_climbing_state_.clear();
    LOG_DEBUG("清除所有攀爬检测记录");
}