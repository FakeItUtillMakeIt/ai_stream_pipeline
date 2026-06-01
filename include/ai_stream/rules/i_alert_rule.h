// include/ai_stream/rules/i_alert_rule.h
#pragma once

#include "ai_stream/core/packet.h"
#include "utils/zone_utils.h"
#include "utils/time_util.h"
#include <string>
#include <vector>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>

namespace ai_stream
{
    namespace rules
    {
        typedef std::vector<PixelPoint> ZonePoints;

        /**
         * @brief 告警规则接口（不继承 Node）
         */
        class IAlertRule
        {
        public:
            virtual ~IAlertRule() = default;

            virtual bool initialize(const nlohmann::json &config) = 0;

            virtual RuleStatus process(
                std::shared_ptr<core::InferenceResultPacket> packet,
                AlertResult &alert_result,
                int64_t current_time_ms=0) = 0;

            virtual void setName(const std::string& name) {alert_name_=name;}
            virtual std::string getName() const { return alert_name_.empty()?alertTypeMap[getType()]:alert_name_; };
            virtual AlertType getType() const = 0;
            virtual void reset() = 0;
            virtual nlohmann::json getStatistics() const = 0;
            virtual RuleStatus rule_logic(
                const std::shared_ptr<core::InferenceResultPacket> packet,
                uint8_t zone_no, ZonePoints zone_points) = 0;
            std::string getTypeName(){return alertTypeMap[getType()];}

            std::map<uint8_t, AlertEvent> zone_alert_map_; // 区域告警事件

            uint8_t global_zone_no_ = 0;
            uint8_t max_disappear_count_ = 5; // 告警事件最大非更新帧数，超过则认为事件结束
            uint64_t alert_duration_ms_ = 1000; // 告警持续时间，超过则认为事件开始

            std::map<uint8_t, ZonePoints> intrusion_zones_;// 区域列表
            std::map<uint8_t, ZonePoints> valid_intrusion_zones_;
            mutable std::mutex mutex_;
            std::string alert_name_;
        };

        using AlertRulePtr = std::shared_ptr<IAlertRule>;

    } // namespace rules
} // namespace ai_stream