// src/rules/alert/absence_rule.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream {
namespace rules {

struct WorkZone {
    int id;
    std::string name;
    float x, y, w, h;
    bool is_rectangle = true;
    std::vector<std::pair<float, float>> polygon;
    int warning_time_sec = 10;
    int alarm_time_sec = 30;
};

class AbsenceRule : public IAlertRule {
public:
    AbsenceRule() = default;
    
    bool initialize(const nlohmann::json& config) override;
    std::vector<AlertEvent> process(
        std::shared_ptr<core::InferenceResultPacket> packet,
        int64_t current_time_ms) override;
    std::string getName() const override { return alertTypeMap[getType()]; }
    AlertType getType() const override { return AlertType::ABSENCE; }
    void reset() override;
    nlohmann::json getStatistics() const override;

private:
    struct PersonState {
        int track_id;
        int zone_id;
        bool in_zone;
        int64_t leave_start_ms;
        bool warning_sent;
        bool alarm_sent;
    };

    bool isInZone(const core::InferenceResultPacket::BBox& box, const WorkZone& zone);
    bool isPointInPolygon(float px, float py, 
                          const std::vector<std::pair<float, float>>& polygon);

    std::vector<WorkZone> zones_;
    std::unordered_map<int, PersonState> person_states_;
    mutable std::mutex mutex_;
    int total_alarms_ = 0;
};

} // namespace rules
} // namespace ai_stream