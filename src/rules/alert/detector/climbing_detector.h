#ifndef CLAMING_DETECTOR_H
#define CLAMING_DETECTOR_H

#include "3rd_party/log_mgr/log_mgr.h"
#include "ai_stream/core/packet.h"
#include <map>
#include <vector>
#include <deque>
#include <memory>
#include <string>
#include <cmath>


class ClimbingDetector {
public:
    ClimbingDetector();
    
    /**
     * 更新检测结果
     * @param track_id 跟踪ID
     * @param person_box 人体边界框
     * @param frame_id 帧ID（用于计算时间差）
     * @return 是否处于攀爬状态
     */
    bool updateDetection(int track_id, const ai_stream::core::InferenceResultPacket::BBox &person_box, int frame_id);
    
    /**
     * 获取指定track_id的攀爬状态
     */
    bool getClimbingState(int track_id) const;
    
    /**
     * 获取攀爬状态的详细信息
     */
    struct ClimbingDetails {
        bool is_climbing;
        int climbing_frames;
        float avg_vertical_speed;
    };
    std::unique_ptr<ClimbingDetails> getClimbingDetails(int track_id) const;
    
    /**
     * 清理不活跃的跟踪记录
     * @param active_track_ids 活跃的track_id集合
     * @param max_age 最大保留帧数（未使用，保留接口兼容）
     */
    void cleanupOldTracks(const std::vector<int>& active_track_ids, int max_age = 50);
    
    /**
     * 手动清除指定track_id的记录
     */
    void removeTrack(int track_id);
    
    /**
     * 清除所有记录
     */
    void clearAll();

private:
    // 历史记录点
    struct HistoryPoint {
        float center_x;
        float center_y;
        float bottom_y;
        float height;
        int frame_id;
        
        HistoryPoint(float cx, float cy, float by, float h, int fid)
            : center_x(cx), center_y(cy), bottom_y(by), height(h), frame_id(fid) {}
    };
    
    // 轨道状态
    struct TrackState {
        std::deque<HistoryPoint> history;           // 位置历史
        std::deque<float> vertical_speeds;          // 垂直速度历史
        bool is_climbing;                           // 是否处于攀爬状态
        int climbing_frames;                        // 连续攀爬状态帧计数
        int non_climbing_frames;                    // 连续非攀爬状态帧计数
        
        TrackState() : is_climbing(false), climbing_frames(0), non_climbing_frames(0) {}
    };
    
    // 配置参数
    float vertical_speed_threshold_;    // 垂直速度阈值（像素/帧）
    float ascent_speed_threshold_;      // 上升速度阈值（像素/帧）
    int window_size_;                   // 滑动窗口大小
    int history_length_;                // 保留最近N帧的历史
    int required_climbing_frames_;      // 需要连续多少帧满足条件才标记为攀爬
    float height_change_ratio_;         // 高度变化比例阈值
    
    // 存储所有轨道的状态
    std::map<int, TrackState> track_climbing_state_;
};

#endif // CLAMING_DETECTOR_H