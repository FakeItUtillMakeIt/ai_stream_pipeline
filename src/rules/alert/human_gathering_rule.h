// src/rules/alert/human_gathering.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream
{
    namespace rules
    {

        class HumanGatheringRule : public IAlertRule
        {
        public:
            HumanGatheringRule();

            bool initialize(const nlohmann::json &config) override;
            RuleStatus process(
                std::shared_ptr<core::InferenceResultPacket> packet,
                AlertResult &alert_result,
                int64_t current_time_ms) override;

            std::string getName() const override { return alertTypeMap[getType()]; }
            AlertType getType() const override { return AlertType::HUMAN_GATHERING; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet, uint8_t zone_no, ZonePoints zone_points) override;

        private:
            std::map<uint8_t, int> gathering_thresh_map_; // 区域聚集人数阈值
        };
    }
}