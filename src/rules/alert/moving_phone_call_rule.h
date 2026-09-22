// src/rules/alert/moving_phone_call_rule.h
#pragma once

#include "detector/moving_phonecall_detector.h"
#include "alert_rule_base.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream { 
namespace rules {
    class MovingPhoneCallRule : public AlertRuleBase {
    public:
        MovingPhoneCallRule() ;
        
        bool initialize(const nlohmann::json& config) override;
        void reset() override;

        AlertType getType() const override{ return AlertType::PHONE_CALL; };
        AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; };
        nlohmann::json getStatistics() const override;

    protected:
        void onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet) override;

    private:
        RuleStatus rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points) override;

    private:
            MovingPhonecallDetector moving_pc_detector_;
            // 每帧只运行一次检测器，结果缓存供各 zone 复用
            std::vector<int> last_moving_phonecall_track_ids_;
    };
}
}