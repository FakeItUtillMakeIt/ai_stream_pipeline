// src/rules/alert/phone_call_rule.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream { 
namespace rules {
    class PhoneCallRule : public IAlertRule {
    public:
        PhoneCallRule() ;
        
        bool initialize(const nlohmann::json& config) override;
        RuleStatus process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult& alert_result,
            int64_t current_time_ms) override;
        void reset() override;

        AlertType getType() const override{ return AlertType::PHONE_CALL; };
        nlohmann::json getStatistics() const override;

    private:
        RuleStatus rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points) override;
    };
}
}