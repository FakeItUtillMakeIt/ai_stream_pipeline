// src/rules/alert/absence_rule.cpp
#include "absence_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace rules {

bool AbsenceRule::initialize(const nlohmann::json& config) {
    try {
        if (config.contains("zones") && config["zones"].is_array()) {
            for (const auto& zc : config["zones"]) {
                WorkZone zone;
                zone.id = zc.value("id", 0);
                zone.name = zc.value("name", "未命名");
                zone.warning_time_sec = zc.value("warning_time", 10);
                zone.alarm_time_sec = zc.value("alarm_time", 30);
                
                if (zc.contains("x")) {
                    zone.x = zc["x"]; zone.y = zc["y"];
                    zone.w = zc["w"]; zone.h = zc["h"];
                    zone.is_rectangle = true;
                } else if (zc.contains("polygon")) {
                    zone.is_rectangle = false;
                    for (const auto& p : zc["polygon"]) {
                        zone.polygon.push_back({p[0], p[1]});
                    }
                }
                zones_.push_back(zone);
            }
        }
        LOG_INFO_FMT("[AbsenceRule] Initialized with {} zones", zones_.size());
        return true;
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[AbsenceRule] Init failed: {}", e.what());
        return false;
    }
}

std::vector<AlertEvent> AbsenceRule::process(
    std::shared_ptr<core::InferenceResultPacket> packet,
    int64_t current_time_ms) {

    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<AlertEvent> events;

    if (!packet || zones_.empty()) return events;

    for (const auto& det : packet->detections) {
        if (det.track_id < 0 || det.class_name != "person") continue;

        for (const auto& zone : zones_) {
            bool in_zone = isInZone(det, zone);
            auto it = person_states_.find(det.track_id);

            if (it == person_states_.end()) {
                person_states_[det.track_id] = {
                    det.track_id, zone.id, in_zone,
                    in_zone ? 0 : current_time_ms, false, false
                };
            } else {
                auto& s = it->second;
                if (!s.in_zone && in_zone) {
                    s.in_zone = true;
                    s.warning_sent = false;
                    s.alarm_sent = false;
                } else if (s.in_zone && !in_zone) {
                    s.in_zone = false;
                    s.leave_start_ms = current_time_ms;
                }
            }
        }
    }

    // 检查超时
    for (auto& [tid, state] : person_states_) {
        if (state.in_zone) continue;
        int dur = (current_time_ms - state.leave_start_ms) / 1000;

        WorkZone* zone = nullptr;
        for (auto& z : zones_) {
            if (z.id == state.zone_id) { zone = &z; break; }
        }
        if (!zone) continue;

        if (dur >= zone->alarm_time_sec && !state.alarm_sent) {
            state.alarm_sent = true;
            state.warning_sent = true;
            AlertEvent e;
            e.timestamp_ms = current_time_ms;
            e.level = AlertLevel::CRITICAL;
            e.rule_name = getName();
            e.rule_type = getName();
            e.stream_id = packet->stream_id;
            e.track_ids = {tid};
            e.description = fmt::format("人员{}离岗{}秒", tid, dur);
            e.extra_data = {{"zone", zone->name}, {"duration", dur}};
            events.push_back(e);
            total_alarms_++;
        } else if (dur >= zone->warning_time_sec && !state.warning_sent) {
            state.warning_sent = true;
            AlertEvent e;
            e.timestamp_ms = current_time_ms;
            e.level = AlertLevel::WARNING;
            e.rule_name = getName();
            e.rule_type = getName();
            e.stream_id = packet->stream_id;
            e.track_ids = {tid};
            e.description = fmt::format("人员{}离岗{}秒(预警)", tid, dur);
            e.extra_data = {{"zone", zone->name}, {"duration", dur}};
            events.push_back(e);
        }
    }
    return events;
}

void AbsenceRule::reset() {
    std::lock_guard<std::mutex> lock(mutex_);
    person_states_.clear();
    total_alarms_ = 0;
}

nlohmann::json AbsenceRule::getStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return {{"total_alarms", total_alarms_}, {"active_persons", person_states_.size()}};
}

bool AbsenceRule::isInZone(const core::InferenceResultPacket::BBox& box, const WorkZone& zone) {
    float cx = box.x + box.w / 2;
    float cy = box.y + box.h;  // 脚底中心
    if (zone.is_rectangle) {
        return cx >= zone.x && cx <= zone.x + zone.w &&
               cy >= zone.y && cy <= zone.y + zone.h;
    }
    return isPointInPolygon(cx, cy, zone.polygon);
}

bool AbsenceRule::isPointInPolygon(float px, float py,
                                    const std::vector<std::pair<float, float>>& poly) {
    int n = poly.size();
    if (n < 3) return false;
    bool inside = false;
    for (int i = 0, j = n - 1; i < n; j = i++) {
        if (((poly[i].second > py) != (poly[j].second > py)) &&
            (px < (poly[j].first - poly[i].first) * (py - poly[i].second) /
             (poly[j].second - poly[i].second) + poly[i].first)) {
            inside = !inside;
        }
    }
    return inside;
}

REGISTER_ALERT_RULE("absence", AbsenceRule)

} // namespace rules
} // namespace ai_stream