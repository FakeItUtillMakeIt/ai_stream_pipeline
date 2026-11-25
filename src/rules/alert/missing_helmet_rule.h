// src/rules/alert/missing_helmet_rule.h
#pragma once

#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream { 
namespace rules {
    class MissingHelmetRule : public AlertRuleBase {
    public:
        MissingHelmetRule() ;
        
        bool initialize(const nlohmann::json& config) override;
        void reset() override;

        AlertType getType() const override{ return AlertType::MISSING_HELMET; };
        AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; };
        nlohmann::json getStatistics() const override;

    private:
        RuleStatus rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points) override;

        // 人体框最小像素高度（1080p 下过滤远处/抖动小目标；可由 config 覆盖）
        int min_person_h_ = 60;
    };
}
}