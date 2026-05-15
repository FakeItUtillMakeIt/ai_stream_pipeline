#ifndef MOVING_PHONECALL_DETECTOR_H
#define MOVING_PHONECALL_DETECTOR_H

#include "utils/zone_utils.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <deque>
#include <memory>
#include <unordered_set>


struct TrackState {
    std::deque<std::tuple<PixelPoint, int>> history;  // center_x, center_y, frame_id
    std::deque<double> speeds;                             // 速度历史
    bool is_moving;                                        // 是否移动
    int moving_frames;                                     // 连续移动帧计数
    int static_frames;                                     // 连续静止帧计数
    
    TrackState() : is_moving(false), moving_frames(0), static_frames(0) {}
};

class MovingPhonecallDetector {
public:
    /**
     * 构造函数
     * @param video_path 视频路径（用于日志记录）
     */
    explicit MovingPhonecallDetector(const std::string& video_path = "");
    
    /**
     * 更新跟踪状态并判断是否移动
     * @param track_id 跟踪ID
     * @param person_box 人体边界框 [x1, y1, x2, y2]
     * @param frame_id 帧ID
     * @return 是否处于移动状态
     */
    bool update_track(int track_id, const std::vector<double>& person_box, int frame_id);
    
    /**
     * 获取指定track_id的移动状态
     * @param track_id 跟踪ID
     * @return 是否移动
     */
    bool get_moving_state(int track_id) const;
    
    /**
     * 清理不活跃的跟踪记录
     * @param active_track_ids 活跃的track_id集合
     * @param max_age 最大保留时长（帧数），未使用但保留接口兼容性
     */
    void cleanup_old_tracks(const std::unordered_set<int>& active_track_ids, int max_age = 50);
    
private:
    std::string video_path_;                                    // 视频路径
    std::unordered_map<int, TrackState> track_moving_state_;   // 跟踪状态映射
    
    double moving_speed_threshold_;      // 移动速度阈值（像素/帧）
    double static_speed_threshold_;      // 静止速度阈值（像素/帧）
    int window_size_;                    // 滑动窗口大小
    int history_length_;                 // 保留最近N帧的历史
    int required_moving_frames_;         // 需要连续多少帧平均速度超过阈值才标记为移动
    
    /**
     * 计算两点之间的欧氏距离
     */
    double calculate_distance(PixelPoint p1, PixelPoint p2) const;
    
    /**
     * 计算边界框中心点
     */
    std::pair<double, double> calculate_center(const std::vector<double>& person_box) const;
};

#endif // MOVING_PHONECALL_DETECTOR_H