// src/rules/alert/person_loitering_rule.h
#pragma once

#include "alert_rule_base.h"
#include <cstdint>
#include <unordered_map>

namespace ai_stream
{
    namespace rules
    {
        /**
         * @brief 人员徘徊检测
         *
         * 与"人员入侵"(进入即报)不同：按 track_id 统计每个人在区域内的**连续滞留时长**，
         * 停留超过 loiter_duration_ms 才判为徘徊。需要稳定的跟踪 ID。
         */
        class PersonLoiteringRule : public AlertRuleBase
        {
        public:
            PersonLoiteringRule();

            bool initialize(const nlohmann::json &config) override;

            AlertType getType() const override { return AlertType::PERSON_LOITERING; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet,
                                  uint8_t zone_no, ZonePoints zone_points) override;

            // (zone_no<<32 | track_id) -> 首次进入该区域的时刻(ms)
            std::unordered_map<uint64_t, int64_t> first_in_zone_ms_;
            // (zone_no<<32 | track_id) -> 最近一次在区域内被看到的时刻(ms)，用于离开后清理
            std::unordered_map<uint64_t, int64_t> last_seen_ms_;

            std::string target_class_ = "person";
            int64_t loiter_duration_ms_ = 10000;   // 滞留多久算徘徊
            int64_t leave_grace_ms_ = 2000;        // 离开区域后保留滞留记录的宽限(容忍漏检)
        };
    }
}
