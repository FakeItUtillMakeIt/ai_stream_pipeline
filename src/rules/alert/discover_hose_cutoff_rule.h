// src/rules/alert/discover_hose_cutoff_rule.h

#pragma once

#include "alert_rule_base.h"
#include <unordered_map>

namespace ai_stream
{
    namespace rules
    {

        class DiscoverHoseCutoffRule : public AlertRuleBase
        {
        public:
            DiscoverHoseCutoffRule();

            bool initialize(const nlohmann::json &config) override;

            AlertType getType() const override { return AlertType::DISCOVER_HOSE_CUTOFF; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_SCENE_RECOGNITION; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet, uint8_t zone_no, ZonePoints zone_points) override;

        };
    }
}