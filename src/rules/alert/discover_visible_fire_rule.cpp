// src/rules/alert/discover_visible_fire_rule.cpp
#include "discover_visible_fire_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        DiscoverVisibleFireRule::DiscoverVisibleFireRule() 
        {
            LOG_INFO("DiscoverVisibleFireRule::DiscoverVisibleFireRule()");
        }
        bool DiscoverVisibleFireRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("DiscoverVisibleFireRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("DiscoverVisibleFireRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void DiscoverVisibleFireRule::reset()
        {
            LOG_INFO_FMT("DiscoverVisibleFireRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json DiscoverVisibleFireRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("DiscoverVisibleFireRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus DiscoverVisibleFireRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("DiscoverVisibleFireRule::rule_logic()");
            int fire_count = 0;
            std::vector<int> fire_track_ids;
            for (const auto &box : packet->detections)
            {
                if (box.class_name != "fire")
                { 
                    continue;
                } 
                bool in_zone = zone_points.empty() ? true : ZoneValidator::pointInPolygon(PixelPoint(box.x + box.w / 2, box.y + box.h / 2), zone_points);
                if (in_zone)
                {
                    fire_count++;
                    fire_track_ids.push_back(box.track_id);
                }
            }
            if (fire_count < 1)
                return RuleStatus::RULE_STATUS_OK;
                
            updateZoneEvent(zone_no, packet, fire_track_ids);

            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("discover_visible_fire", DiscoverVisibleFireRule)
    }
}
