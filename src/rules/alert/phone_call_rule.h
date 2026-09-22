// src/rules/alert/phone_call_rule.h
#pragma once

#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream { 
namespace rules {
    class PhoneCallRule : public AlertRuleBase {
    public:
        PhoneCallRule() ;
        
        bool initialize(const nlohmann::json& config) override;
        void reset() override;

        AlertType getType() const override{ return AlertType::PHONE_CALL; };
        AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; };
        nlohmann::json getStatistics() const override;

    private:
        RuleStatus rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points) override;
    };
}
}