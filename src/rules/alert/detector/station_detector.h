#ifndef STATION_DETECTOR_H
#define STATION_DETECTOR_H

#include "ai_stream/core/packet.h"
#include "utils/zone_utils.h"
#include <map>
#include <vector>
#include <deque>
#include <chrono>
#include <cmath>
#include <limits>

struct StationRegion {
    int track_id;                           // 关联的跟踪ID
    std::vector<PixelPoint> region;         // 岗位区域（多边形顶点）
    PixelPoint center;                      // 区域中心
    long long created_time;                 // 创建时间
    bool is_valid;                          // 是否有效
    
    StationRegion() : track_id(-1), created_time(0), is_valid(false) {}
    
    StationRegion(int id, const std::vector<PixelPoint>& poly, long long time)
        : track_id(id), region(poly), created_time(time), is_valid(true) {
        // 计算中心点
        if (!region.empty()) {
            float sum_x = 0.0f, sum_y = 0.0f;
            for (const auto& p : region) {
                sum_x += p.x;
                sum_y += p.y;
            }
            center.x = sum_x / region.size();
            center.y = sum_y / region.size();
        }
    }
    
    // 获取边界框（用于快速判断）
    void getBoundingBox(float& min_x, float& max_x, float& min_y, float& max_y) const {
        if (region.empty()) {
            min_x = max_x = min_y = max_y = 0;
            return;
        }
        min_x = max_x = region[0].x;
        min_y = max_y = region[0].y;
        for (const auto& p : region) {
            min_x = std::min(min_x, p.x);
            max_x = std::max(max_x, p.x);
            min_y = std::min(min_y, p.y);
            max_y = std::max(max_y, p.y);
        }
    }
};

// 辅助结构：矩形（用于内部计算）
struct Rect2f {
    float x, y, width, height;
    
    Rect2f() : x(0), y(0), width(0), height(0) {}
    Rect2f(float _x, float _y, float _w, float _h) : x(_x), y(_y), width(_w), height(_h) {}
    
    float left() const { return x; }
    float right() const { return x + width; }
    float top() const { return y; }
    float bottom() const { return y + height; }
    float centerX() const { return x + width / 2; }
    float centerY() const { return y + height / 2; }
    PixelPoint center() const { return {centerX(), centerY()}; }
};

class StationDetector {
public:
    // 构造函数
    StationDetector();
    
    // 带参数构造函数
    StationDetector(int min_stay_duration,      // 最小停留时长(毫秒)
                   float position_threshold,    // 位置变化阈值(像素)
                   int min_history_size,        // 最小历史点数
                   float region_expand_ratio);  // 区域扩展比例

    std::vector<StationRegion> getAllStations(std::vector<ai_stream::core::InferenceResultPacket::BBox>& all_detections);
    
    // 核心方法：更新轨迹并获取岗位区域
    // 返回: true-已检测到岗位区域, false-还在学习中
    bool updateAndGetStation(int track_id, 
                            const Rect2f& bbox, 
                            long long timestamp,
                            StationRegion& out_region);
    
    // 简化方法：只更新，不返回区域
    void update(int track_id, const Rect2f& bbox, long long timestamp);
    
    // 获取指定人员的岗位区域（如果已检测到）
    bool getStationRegion(int track_id, StationRegion& out_region);
    
    // 检查是否已检测到岗位区域
    bool hasStationRegion(int track_id);
    
    // 强制重置指定人员的岗位区域（重新学习）
    void resetStation(int track_id);
    
    // 清除所有岗位区域
    void clearAllStations();
    
    // 保存/加载岗位区域
    bool saveStationsToFile(const std::string& filename);
    bool loadStationsFromFile(const std::string& filename);
    
    // 参数设置
    void setMinStayDuration(int duration_ms);
    void setPositionThreshold(float threshold);
    void setRegionExpandRatio(float ratio);
    
    // 获取统计信息
    size_t getStationCount() const { return m_stations.size(); }
    std::vector<int> getAllTrackIds() const;

private:
    // 人员轨迹数据
    struct PersonTrack {
        int id;
        std::deque<Rect2f> history;    // 边界框历史
        long long stay_start_time;      // 开始停留时间
        PixelPoint last_position;       // 上一帧位置
        bool has_station;               // 是否已有岗位区域
        StationRegion station;          // 岗位区域
        
        PersonTrack() : id(-1), stay_start_time(0), has_station(false) {
            last_position.x = last_position.y = 0;
        }
    };
    
    // 内部方法
    PixelPoint getStablePoint(const Rect2f& bbox);
    std::vector<PixelPoint> calculateRegion(const std::deque<Rect2f>& history);
    bool isPositionStable(const PixelPoint& current, const PixelPoint& last);
    long long getCurrentTime();
    void cleanExpiredTracks(long long current_time);
    
    // 辅助几何方法
    static bool pointInPolygon(const PixelPoint& pt, const std::vector<PixelPoint>& polygon);
    static std::vector<PixelPoint> rectToPolygon(const Rect2f& rect);
    static Rect2f polygonToBoundingBox(const std::vector<PixelPoint>& polygon);
    
    // 成员变量
    std::map<int, PersonTrack> m_tracks;      // 所有跟踪轨迹
    std::map<int, StationRegion> m_stations;  // 已检测到的岗位区域
    
    // 配置参数
    int m_min_stay_duration;       // 默认5000ms
    float m_position_threshold;    // 默认30像素
    int m_min_history_size;        // 默认50帧
    float m_region_expand_ratio;   // 默认0.2
    long long m_track_expire_time; // 默认10000ms
};

#endif // STATION_DETECTOR_H