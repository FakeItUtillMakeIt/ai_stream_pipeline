// src/rules/alert/climbing_rule.h
#pragma once

#include "detector/climbing_detector.h"
#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream
{
    namespace rules
    {

        class ClimbingRule : public AlertRuleBase
        {
        public:
            ClimbingRule();

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
            ClimbingDetector climbing_detector_;
            // 每帧只运行一次检测器，结果缓存供各 zone 复用
            bool last_is_climbing_ = false;
            std::vector<int> last_climb_track_ids_;
        };
    }
}