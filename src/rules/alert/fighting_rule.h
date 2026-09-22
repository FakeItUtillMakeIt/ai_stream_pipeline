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
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_SCENE_RECOGNITION; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet, uint8_t zone_no, ZonePoints zone_points) override;

        private:
            FightingDetector fighting_detector_;
            // 每帧只运行一次检测器，结果缓存供各 zone 复用
            bool last_is_fighting_ = false;
            std::vector<int> last_fight_track_ids_;
        };
    }
}