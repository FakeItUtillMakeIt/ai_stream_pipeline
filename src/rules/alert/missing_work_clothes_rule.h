// src/rules/alert/missing_work_clothes_rule.h
#pragma once

#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream { 
namespace rules {
    class MissingWorkClothesRule : public AlertRuleBase {
    public:
        MissingWorkClothesRule() ;
        
        bool initialize(const nlohmann::json& config) override;
        void reset() override;

        AlertType getType() const override{ return AlertType::MISSING_WORK_CLOTHES; };
        AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_SAFETY_ITEM; };
        nlohmann::json getStatistics() const override;

    private:
        RuleStatus rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points) override;

        // 可配置判定参数（1080p 默认值）
        float cover_ratio_ = 0.3f;   // 工作服框与人体框重叠面积 / 人体框面积 >= 此值视为"已穿"
        int   min_person_h_ = 80;    // 人体框最小像素高度，低于此跳过（远处抖动/误检）
        float min_conf_ = 0.0f;      // 参与判定的检测框最低置信度（默认沿用管线阈值）
    };
}
}