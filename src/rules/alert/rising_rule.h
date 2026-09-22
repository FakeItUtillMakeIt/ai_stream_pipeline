// src/rules/alert/rising_rule.h
#pragma once

#include "detector/person_box_rising_detector.h"
#include "ai_stream/rules/i_alert_rule.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream
{
    namespace rules
    {

        class RisingRule : public IAlertRule
        {
        public:
            RisingRule();

            bool initialize(const nlohmann::json &config) override;
            RuleStatus process(
                std::shared_ptr<core::InferenceResultPacket> packet,
                AlertResult &alert_result,
                int64_t current_time_ms) override;

            AlertType getType() const override { return AlertType::CLIMBING; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet, uint8_t zone_no, ZonePoints zone_points) override;

        private:
            PersonBoxRisingDetector rising_detector_;
            // 每帧只运行一次检测器，结果缓存供各 zone 的 rule_logic 复用，
            // 避免多区域配置下重复推进检测器状态机
            RisingResult last_rising_result_;
        };
    }
}