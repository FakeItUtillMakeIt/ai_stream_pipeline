// src/rules/alert/absence_rule.h
#pragma once

#include "detector/station_detector.h"
#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream { 
namespace rules {
    class AbsenceRule : public AlertRuleBase {
    public:
        AbsenceRule() ;
        
        bool initialize(const nlohmann::json& config) override;
        void reset() override;

        AlertType getType() const override{ return AlertType::ABSENCE; };
        AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; };
        nlohmann::json getStatistics() const override;

    protected:
        void onPreProcess(const std::shared_ptr<core::InferenceResultPacket>& packet) override;

    private:
        RuleStatus rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points) override;

    private:
        StationDetector station_detector_;// 岗位检测器
        std::vector<StationRegion> all_station_regions_; // 岗位区域列表
        std::map<int,int> m_absent_counter_map_;
        int absent_threshold_;
    };
}
}