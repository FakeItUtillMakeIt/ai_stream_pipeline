// src/rules/alert/missing_work_clothes_rule.h
#pragma once

#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream { 
namespace rules {
    class MissingWorkClothesRule : public AlertRuleBase {
    public:
        MissingWorkClothesRule() ;
        
        bool initialize(const nlohmann::json& config) override;
        void reset() override;

        AlertType getType() const override{ return AlertType::MISSING_WORK_CLOTHES; };
        AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_SAFETY_ITEM; };
        nlohmann::json getStatistics() const override;

    private:
        RuleStatus rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points) override;
    };
}
}