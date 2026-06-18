// src/rules/alert/climbing_rule.h
#pragma once

#include "detector/climbing_detector.h"
#include "ai_stream/rules/i_alert_rule.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream
{
    namespace rules
    {

        class ClimbingRule : public IAlertRule
        {
        public:
            ClimbingRule();

            bool initialize(const nlohmann::json &config) override;
            RuleStatus process(
                std::shared_ptr<core::InferenceResultPacket> packet,
                AlertResult &alert_result,
                int64_t current_time_ms) override;

            AlertType getType() const override { return AlertType::HUMAN_GATHERING; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet, uint8_t zone_no, ZonePoints zone_points) override;

        private:
            ClimbingDetector climbing_detector_;
        };
    }
}