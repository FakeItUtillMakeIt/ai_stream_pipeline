// src/rules/alert/person_instrusion_rule.cpp
#include "person_intrusion_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        PersonIntrusionRule::PersonIntrusionRule()
        {
            LOG_INFO("PersonIntrusionRule::PersonIntrusionRule()");
        }

        bool PersonIntrusionRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("PersonIntrusionRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("PersonIntrusionRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void PersonIntrusionRule::reset()
        {
            LOG_INFO_FMT("PersonIntrusionRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json PersonIntrusionRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("PersonIntrusionRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus PersonIntrusionRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("PersonIntrusionRule::rule_logic()");
            bool person_in_zone = false;
            std::vector<int> person_track_ids;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name != "person")
                {
                    continue;
                }
                bool in_zone = zone_points.empty() ? true : ZoneValidator::pointInPolygon(PixelPoint(detection.x + detection.w / 2, detection.y + detection.h / 2), zone_points);
                if (in_zone)
                {
                    person_in_zone = true;
                    person_track_ids.push_back(detection.track_id);
                }
            }
            LOG_DEBUG_FMT("PersonIntrusionRule::rule_logic() zone {} person_in_zone:{},person_track_ids size:{},packet_time:{}",
                          zone_no, person_in_zone, person_track_ids.size(), packet->timestamp_ms);
            // 更新zone_alert_map_
            if (!person_in_zone)
                return RuleStatus::RULE_STATUS_OK;
            updateZoneEvent(zone_no, packet, person_track_ids);

            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("person_intrusion", PersonIntrusionRule)
    }
}
