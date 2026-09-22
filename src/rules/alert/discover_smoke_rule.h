// src/rules/alert/discover_smoking_rule.h

#pragma once

#include "alert_rule_base.h"
#include <unordered_map>

namespace ai_stream
{
    namespace rules
    {

        class DiscoverSmokeRule : public AlertRuleBase
        {
        public:
            DiscoverSmokeRule();

            bool initialize(const nlohmann::json &config) override;

            AlertType getType() const override { return AlertType::DISCOVER_SMOKE; }
            AlertItemType getAlertItemType() const override { return alertItemTypeMap.at(getType()); }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet, uint8_t zone_no, ZonePoints zone_points) override;

        };
    }
}