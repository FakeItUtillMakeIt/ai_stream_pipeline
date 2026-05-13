#include "station_detector.h"
#include <fstream>
#include <nlohmann/json.hpp>

StationDetector::StationDetector()
    : m_min_stay_duration(5000)
    , m_position_threshold(30.0f)
    , m_min_history_size(50)
    , m_region_expand_ratio(0.2f)
    , m_track_expire_time(10000) {
}

StationDetector::StationDetector(int min_stay_duration,
                                 float position_threshold,
                                 int min_history_size,
                                 float region_expand_ratio)
    : m_min_stay_duration(min_stay_duration)
    , m_position_threshold(position_threshold)
    , m_min_history_size(min_history_size)
    , m_region_expand_ratio(region_expand_ratio)
    , m_track_expire_time(10000) {
}

std::vector<StationRegion> StationDetector::getAllStations(std::vector<ai_stream::core::InferenceResultPacket::BBox>& all_detections) 
{
    std::vector<StationRegion> stations;
    for(const auto& each_det : all_detections)
    {
        if(each_det.class_id == 0) // 只考虑人员检测结果
        {
            // 将Detection转换为Rect2f
            Rect2f bbox(each_det.x, each_det.y, each_det.w, each_det.h);
            // 更新每个目标
            update(each_det.track_id, bbox, getCurrentTime());
            // 获取岗位区域
            StationRegion region;
            if (getStationRegion(each_det.track_id, region)) {
                stations.push_back(region);
            }
        }
    }
    return stations;
}

bool StationDetector::updateAndGetStation(int track_id,
                                          const Rect2f& bbox,
                                          long long timestamp,
                                          StationRegion& out_region) {
    update(track_id, bbox, timestamp);
    return getStationRegion(track_id, out_region);
}

void StationDetector::update(int track_id, const Rect2f& bbox, long long timestamp) {
    if (track_id < 0 || bbox.width <= 0 || bbox.height <= 0) {
        return;
    }
    
    PixelPoint current_pos = getStablePoint(bbox);
    
    // 获取或创建轨迹
    auto& track = m_tracks[track_id];
    if (track.id == -1) {
        track.id = track_id;
        track.last_position = current_pos;
        track.stay_start_time = timestamp;
    }
    
    // 如果已经有岗位区域，不需要再学习
    if (track.has_station) {
        track.last_position = current_pos;
        track.history.push_back(bbox);
        // 限制历史大小
        while (track.history.size() > 1000) {
            track.history.pop_front();
        }
        cleanExpiredTracks(timestamp);
        return;
    }
    
    // 检查位置稳定性
    bool is_stable = isPositionStable(current_pos, track.last_position);
    
    if (is_stable) {
        // 位置稳定，开始或继续计时
        if (track.stay_start_time == 0) {
            track.stay_start_time = timestamp;
        } else if (!track.has_station) {
            long long stay_duration = timestamp - track.stay_start_time;
            // 达到停留时长阈值，生成岗位区域
            if (stay_duration >= m_min_stay_duration && 
                (int)track.history.size() >= m_min_history_size) {
                
                std::vector<PixelPoint> region = calculateRegion(track.history);
                track.station = StationRegion(track_id, region, timestamp);
                track.has_station = true;
                m_stations[track_id] = track.station;
            }
        }
    } else {
        // 位置变化，重置计时
        track.stay_start_time = 0;
    }
    
    // 记录历史
    track.history.push_back(bbox);
    track.last_position = current_pos;
    
    // 限制历史大小
    while (track.history.size() > 1000) {
        track.history.pop_front();
    }
    
    // 清理过期轨迹
    cleanExpiredTracks(timestamp);
}

bool StationDetector::getStationRegion(int track_id, StationRegion& out_region) {
    auto it = m_stations.find(track_id);
    if (it != m_stations.end() && it->second.is_valid) {
        out_region = it->second;
        return true;
    }
    return false;
}

bool StationDetector::hasStationRegion(int track_id) {
    auto it = m_stations.find(track_id);
    return it != m_stations.end() && it->second.is_valid;
}

void StationDetector::resetStation(int track_id) {
    auto it = m_tracks.find(track_id);
    if (it != m_tracks.end()) {
        it->second.has_station = false;
        it->second.stay_start_time = 0;
        it->second.history.clear();
        m_stations.erase(track_id);
    }
}

void StationDetector::clearAllStations() {
    m_tracks.clear();
    m_stations.clear();
}

bool StationDetector::saveStationsToFile(const std::string& filename) {
    nlohmann::json root;
    nlohmann::json stations_array = nlohmann::json::array();
    for (const auto& pair : m_stations) {
        nlohmann::json station_json;
        station_json["track_id"] = pair.first;
        
        // 保存多边形区域
        nlohmann::json polygon_array = nlohmann::json::array();
        for (const auto& pt : pair.second.region) {
            nlohmann::json point;
            point["x"] = pt.x;
            point["y"] = pt.y;
            polygon_array.push_back(point);
        }
        station_json["region"] = polygon_array;
        station_json["center_x"] = pair.second.center.x;
        station_json["center_y"] = pair.second.center.y;
        stations_array.push_back(station_json);
    }
    
    root["stations"] = stations_array;
    root["total_stations"] = (int)m_stations.size();
    
    std::ofstream file(filename);
    if (!file.is_open()) return false;
    file << root.dump(4);

    return true;
}

bool StationDetector::loadStationsFromFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) return false;
    
    nlohmann::json root;
    file >> root;
    
    m_stations.clear();
    const nlohmann::json& stations_array = root["stations"];
    for (const auto& station_json : stations_array) {
        int track_id = station_json["track_id"].get<int>();
        StationRegion station;
        station.track_id = track_id;
        
        // 加载多边形区域
        const nlohmann::json& polygon_array = station_json["region"];
        for (const auto& point_json : polygon_array) {
            PixelPoint pt;
            pt.x = point_json["x"].get<float>();
            pt.y = point_json["y"].get<float>();
            station.region.push_back(pt);
        }
        
        station.center.x = station_json["center_x"].get<float>();
        station.center.y = station_json["center_y"].get<float>();
        station.is_valid = true;
        m_stations[track_id] = station;
        
        // 同时也更新track信息
        auto& track = m_tracks[track_id];
        track.id = track_id;
        track.has_station = true;
        track.station = station;
    }
    return true;
}

void StationDetector::setMinStayDuration(int duration_ms) {
    m_min_stay_duration = duration_ms;
}

void StationDetector::setPositionThreshold(float threshold) {
    m_position_threshold = threshold;
}

void StationDetector::setRegionExpandRatio(float ratio) {
    m_region_expand_ratio = std::max(0.0f, std::min(1.0f, ratio));
}

std::vector<int> StationDetector::getAllTrackIds() const {
    std::vector<int> ids;
    for (const auto& pair : m_stations) {
        ids.push_back(pair.first);
    }
    return ids;
}

// ========== 私有方法 ==========

PixelPoint StationDetector::getStablePoint(const Rect2f& bbox) {
    // 使用边界框底部中心点（更适合人体检测）
    PixelPoint pt;
    pt.x = bbox.x + bbox.width / 2;
    pt.y = bbox.y + bbox.height * 0.85f;
    return pt;
}

std::vector<PixelPoint> StationDetector::calculateRegion(const std::deque<Rect2f>& history) {
    if (history.empty()) return std::vector<PixelPoint>();
    
    float min_x = std::numeric_limits<float>::max();
    float min_y = std::numeric_limits<float>::max();
    float max_x = -std::numeric_limits<float>::max();
    float max_y = -std::numeric_limits<float>::max();
    
    // 计算历史边界框的包围盒
    int start_idx = std::max(0, (int)history.size() - m_min_history_size);
    for (int i = start_idx; i < (int)history.size(); i++) {
        const auto& bbox = history[i];
        min_x = std::min(min_x, bbox.x);
        min_y = std::min(min_y, bbox.y);
        max_x = std::max(max_x, bbox.x + bbox.width);
        max_y = std::max(max_y, bbox.y + bbox.height);
    }
    
    float width = max_x - min_x;
    float height = max_y - min_y;
    float expand_x = width * m_region_expand_ratio;
    float expand_y = height * m_region_expand_ratio;
    
    // 扩展后的矩形
    float final_x = min_x - expand_x;
    float final_y = min_y - expand_y;
    float final_w = width + expand_x * 2;
    float final_h = height + expand_y * 2;
    
    // 将矩形转换为多边形（四个顶点，顺时针方向）
    std::vector<PixelPoint> polygon;
    polygon.push_back({final_x, final_y});                 // 左上
    polygon.push_back({final_x + final_w, final_y});       // 右上
    polygon.push_back({final_x + final_w, final_y + final_h}); // 右下
    polygon.push_back({final_x, final_y + final_h});       // 左下
    
    return polygon;
}

bool StationDetector::isPositionStable(const PixelPoint& current, const PixelPoint& last) {
    float dx = current.x - last.x;
    float dy = current.y - last.y;
    return (dx * dx + dy * dy) < (m_position_threshold * m_position_threshold);
}

long long StationDetector::getCurrentTime() {
    auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
}

void StationDetector::cleanExpiredTracks(long long current_time) {
    auto it = m_tracks.begin();
    while (it != m_tracks.end()) {
        // 如果超过10秒没有更新且没有岗位区域，则移除
        if (!it->second.has_station && 
            it->second.history.empty()) {
            it = m_tracks.erase(it);
        } else {
            ++it;
        }
    }
}

// ========== 辅助几何方法 ==========

bool StationDetector::pointInPolygon(const PixelPoint& pt, const std::vector<PixelPoint>& polygon) {
    if (polygon.size() < 3) return false;
    
    bool inside = false;
    int n = polygon.size();
    
    for (int i = 0, j = n - 1; i < n; j = i++) {
        const auto& p1 = polygon[i];
        const auto& p2 = polygon[j];
        
        if (((p1.y > pt.y) != (p2.y > pt.y)) &&
            (pt.x < (p2.x - p1.x) * (pt.y - p1.y) / (p2.y - p1.y) + p1.x)) {
            inside = !inside;
        }
    }
    
    return inside;
}

std::vector<PixelPoint> StationDetector::rectToPolygon(const Rect2f& rect) {
    std::vector<PixelPoint> polygon;
    polygon.push_back({rect.x, rect.y});                          // 左上
    polygon.push_back({rect.x + rect.width, rect.y});             // 右上
    polygon.push_back({rect.x + rect.width, rect.y + rect.height}); // 右下
    polygon.push_back({rect.x, rect.y + rect.height});            // 左下
    return polygon;
}

Rect2f StationDetector::polygonToBoundingBox(const std::vector<PixelPoint>& polygon) {
    if (polygon.empty()) return Rect2f();
    
    float min_x = polygon[0].x, max_x = polygon[0].x;
    float min_y = polygon[0].y, max_y = polygon[0].y;
    
    for (const auto& pt : polygon) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
    }
    
    return Rect2f(min_x, min_y, max_x - min_x, max_y - min_y);
}