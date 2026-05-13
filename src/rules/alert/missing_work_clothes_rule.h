// src/rules/alert/missing_work_clothes_rule.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream { 
namespace rules {
    class MissingWorkClothesRule : public IAlertRule {
    public:
        MissingWorkClothesRule() ;
        
        bool initialize(const nlohmann::json& config) override;
        RuleStatus process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult& alert_result,
            int64_t current_time_ms) override;
        void reset() override;

        std::string getName() const override{ return alertTypeMap[getType()]; };
        AlertType getType() const override{ return AlertType::MISSING_WORK_CLOTHES; };
        nlohmann::json getStatistics() const override;

    private:
        RuleStatus rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points) override;
    };
}
}