// src/rules/alert/fighting_rule.h

#pragma once

#include "alert_rule_base.h"
#include "detector/fighting_detector.h"
#include <unordered_map>

namespace ai_stream
{
    namespace rules
    {

        class FightingRule : public AlertRuleBase
        {
        public:
            FightingRule();

            bool initialize(const nlohmann::json &config) override;

            AlertType getType() const override { return AlertType::FIGHTING; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_SCENE_RECOGNITION; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        protected:
            void onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet) override;

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