// src/rules/alert/rising_rule.h
#pragma once

#include "detector/person_box_rising_detector.h"
#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream
{
    namespace rules
    {

        class RisingRule : public AlertRuleBase
        {
        public:
            RisingRule();

            bool initialize(const nlohmann::json &config) override;

            AlertType getType() const override { return AlertType::CLIMBING; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        protected:
            void onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet) override;

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