// src/rules/alert/person_instrusion_rule.h
#pragma once

#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream
{
    namespace rules
    {

        class PersonIntrusionRule : public AlertRuleBase
        {
        public:
            PersonIntrusionRule();

            bool initialize(const nlohmann::json &config) override;

            AlertType getType() const override { return AlertType::PERSON_INTRUSION; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet, uint8_t zone_no, ZonePoints zone_points) override;

        private:
        };
    }
}