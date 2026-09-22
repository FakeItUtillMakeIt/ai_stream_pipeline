// src/rules/alert/missing_safety_belt_rule.cpp
#include "missing_safety_belt_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        MissingSafetyBeltRule::MissingSafetyBeltRule()
        {
            LOG_INFO("MissingSafetyBeltRule::MissingSafetyBeltRule()");
        }

        bool MissingSafetyBeltRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("MissingSafetyBeltRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("MissingSafetyBeltRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void MissingSafetyBeltRule::reset()
        {
            LOG_INFO_FMT("MissingSafetyBeltRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json MissingSafetyBeltRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus MissingSafetyBeltRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("MissingSafetyBeltRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            std::vector<core::InferenceResultPacket::BBox> safety_belt_boxes;
            int person_not_safety_belt_count = 0;
            std::vector<int> not_safety_belt_track_ids;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
                }
                else if (detection.class_name == "safety_belt")
                {
                    safety_belt_boxes.push_back(detection);
                }
            }
            for (const auto &person_box : person_boxes)
            {
                bool person_has_safety_belt = false;
                if (safety_belt_boxes.empty())
                {
                    person_not_safety_belt_count++;
                    continue;
                }
                std::vector<PixelPoint> person_zone = {
                    PixelPoint(person_box.x, person_box.y),
                    PixelPoint(person_box.x + person_box.w, person_box.y),
                    PixelPoint(person_box.x + person_box.w, person_box.y + person_box.h),
                    PixelPoint(person_box.x, person_box.y + person_box.h)};
                for (const auto &safety_belt_box : safety_belt_boxes)
                {
                    if (ZoneValidator::pointInPolygon(
                            PixelPoint(safety_belt_box.x + safety_belt_box.w / 2, safety_belt_box.y + safety_belt_box.h / 2),
                            person_zone))
                    {
                        person_has_safety_belt = true;
                        break;
                    }
                }
                if (!person_has_safety_belt)
                {
                    person_not_safety_belt_count++;
                    not_safety_belt_track_ids.push_back(person_box.track_id);
                }
            }
            if (person_not_safety_belt_count > 0)
            {
                updateZoneEvent(zone_no, packet, not_safety_belt_track_ids);
            }
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("missing_safety_belt", MissingSafetyBeltRule)
    }
}