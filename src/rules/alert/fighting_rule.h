// src/rules/alert/fighting_rule.h

#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include "detector/fighting_detector.h"
#include <unordered_map>

namespace ai_stream
{
    namespace rules
    {

        class FightingRule : public IAlertRule
        {
        public:
            FightingRule();

            bool initialize(const nlohmann::json &config) override;
            RuleStatus process(
                std::shared_ptr<core::InferenceResultPacket> packet,
                AlertResult &alert_result,
                int64_t current_time_ms) override;

            AlertType getType() const override { return AlertType::FIGHTING; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet, uint8_t zone_no, ZonePoints zone_points) override;

        private:
            FightingDetector fighting_detector_;
        };
    }
}