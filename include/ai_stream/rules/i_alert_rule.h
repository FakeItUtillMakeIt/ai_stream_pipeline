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

        enum class RuleStatus : uint8_t
        {
            RULE_STATUS_OK = 0,
            RULE_STATUS_FAIL,
            RULE_STATUS_NOT_INITIALIZED,
            RULE_STATUS_NOT_SUPPORTED
        };

        /**
         * @brief 告警级别
         */
        enum class AlertLevel
        {
            INFO = 0,
            WARNING = 1,
            ERROR = 2,
            CRITICAL = 3
        };

        enum class AlertType
        {
            ALERT_UNKNOWN = 0,
            PERSON_INTRUSION = 1,
            MISSING_HELMET = 2,
            MISSING_WORK_CLOTHES = 3,
            PHONE_CALL = 4,
            SMOKING = 5,
            FALL_DOWN = 6,
            MISSING_SAFETY_BELT = 7,
            HUAMN_GATHERING = 8,
            ABSENCE = 9,
            SLEEPING_ON_DUTY = 10,
            CLAMBING = 11,
        };

        std::map<AlertType, std::string> alertTypeMap = {
            {AlertType::PERSON_INTRUSION, "person_intrusion"},
            {AlertType::MISSING_HELMET, "missing_helmet"},
            {AlertType::MISSING_WORK_CLOTHES, "missing_work_clothes"},
            {AlertType::PHONE_CALL, "phone_call"},
            {AlertType::SMOKING, "smoking"},
            {AlertType::FALL_DOWN, "fall_down"},
            {AlertType::MISSING_SAFETY_BELT, "missing_safety_belt"},
            {AlertType::HUAMN_GATHERING, "human_gathering"},
            {AlertType::ABSENCE, "absence"},
            {AlertType::SLEEPING_ON_DUTY, "sleeping_on_duty"},
            {AlertType::CLAMBING, "climbing"}};

        enum class AlertStatus : uint8_t
        {
            ALERT_STATUS_OCCUR = 0,
            ALERT_STATUS_LAST = 1,
            ALERT_STATUS_END = 2,
            ALERT_STATUS_DEFAULT = 3
        };

        /**
         * @brief 告警事件
         */
        struct AlertEvent
        {
            std::string alert_id;
            int64_t detect_ms;
            int64_t duration_ms;
            uint16_t non_update_count;
            uint8_t zone_no;
            AlertLevel level;
            std::string alert_name;
            std::string alert_type;
            std::string description;
            AlertStatus status;
            nlohmann::json extra_data;
            std::vector<int> object_ids;

            nlohmann::json toJson() const
            {
                return {};
            }
            AlertEvent() : detect_ms(0), duration_ms(0), non_update_count(0), zone_no(0), level(AlertLevel::INFO), status(AlertStatus::ALERT_STATUS_DEFAULT) {}
        };

        struct AlertResult
        {
            uint32_t stream_id;
            AlertType alert_type;
            uint8_t alert_count;
            std::vector<AlertEvent> alert_events;
        };

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
                int64_t current_time_ms) = 0;

            virtual std::string getName() const = 0;
            virtual AlertType getType() const = 0;
            virtual void reset() = 0;
            virtual nlohmann::json getStatistics() const = 0;
            virtual RuleStatus rule_logic(
                const std::shared_ptr<core::InferenceResultPacket> packet,
                uint8_t zone_no, ZonePoints zone_points) = 0;

            std::map<uint8_t, AlertEvent> zone_alert_map_; // 区域告警事件

            uint8_t global_zone_no_ = 0;
            uint8_t max_disappear_count_ = 5; // 告警事件最大非更新帧数，超过则认为事件结束
            uint64_t alert_duration_ms_ = 1000; // 告警持续时间，超过则认为事件开始

            std::map<uint8_t, ZonePoints> intrusion_zones_;// 区域列表
            std::map<uint8_t, ZonePoints> valid_intrusion_zones_;
            mutable std::mutex mutex_;
        };

        using AlertRulePtr = std::shared_ptr<IAlertRule>;

    } // namespace rules
} // namespace ai_stream