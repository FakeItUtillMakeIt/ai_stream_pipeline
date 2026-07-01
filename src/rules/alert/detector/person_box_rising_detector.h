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
    bool is_rising;                                 // 是否上升
    int rising_frames;                              // 连续上升帧计数
    int non_rising_frames;                          // 连续非上升帧计数
    int frame_count;                                // 帧计数器
    double peak_y;                                  // 窗口内Y坐标峰值
    double trough_y;                                // 窗口内Y坐标谷值
    
    RisingTrackState() : is_rising(false), rising_frames(0), non_rising_frames(0), 
                          frame_count(0), peak_y(0.0), trough_y(0.0) {}
};

struct RisingResult {
    bool is_rising;                                 // 是否有任何人上升
    std::unordered_map<int, bool> track_rising_states;  // 每个track_id的上升状态
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
    
    bool update_track(int track_id, const core::InferenceResultPacket::BBox& bbox);
    
    double calculate_y_center(const core::InferenceResultPacket::BBox& bbox) const;
    
    double calculate_x_center(const core::InferenceResultPacket::BBox& bbox) const;
    
    double calculate_slope(const std::deque<double>& data) const;
    
    bool check_x_displacement(const std::deque<double>& x_history) const;
    
    bool check_height_stability(const std::deque<double>& h_history) const;
    
    bool check_oscillation(const std::deque<double>& y_history) const;
    
    bool check_sustain_ratio(const RisingTrackState& state) const;
};

#endif // PERSON_BOX_RISING_DETECTOR_H
