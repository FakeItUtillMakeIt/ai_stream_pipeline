// include/ai_stream/rules/i_alert_rule.h
#pragma once

#include "ai_stream/core/packet.h"
#include <string>
#include <vector>
#include <memory>
#include <nlohmann/json.hpp>

namespace ai_stream {
namespace rules {

/**
 * @brief 告警级别
 */
enum class AlertLevel {
    INFO = 0,
    WARNING = 1,
    ERROR = 2,
    CRITICAL = 3
};

enum class AlertType {
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
    {AlertType::CLAMBING, "climbing"}
};

/**
 * @brief 告警事件
 */
struct AlertEvent {
    int64_t timestamp_ms;
    uint32_t stream_id;
    AlertLevel level;
    std::string rule_name;
    std::string rule_type;
    std::string description;
    std::vector<int> track_ids;
    nlohmann::json extra_data;

    nlohmann::json toJson() const {
        return {
            {"timestamp_ms", timestamp_ms},
            {"stream_id", stream_id},
            {"level", static_cast<int>(level)},
            {"rule_name", rule_name},
            {"rule_type", rule_type},
            {"description", description},
            {"track_ids", track_ids},
            {"extra", extra_data}
        };
    }
};

/**
 * @brief 告警规则接口（不继承 Node）
 */
class IAlertRule {
public:
    virtual ~IAlertRule() = default;

    virtual bool initialize(const nlohmann::json& config) = 0;
    
    virtual std::vector<AlertEvent> process(
        std::shared_ptr<core::InferenceResultPacket> packet,
        int64_t current_time_ms) = 0;

    virtual std::string getName() const = 0;
    virtual AlertType getType() const = 0;
    virtual void reset() = 0;
    virtual nlohmann::json getStatistics() const = 0;
};

using AlertRulePtr = std::shared_ptr<IAlertRule>;

} // namespace rules
} // namespace ai_stream