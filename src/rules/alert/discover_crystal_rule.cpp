// src/rules/alert/discover_crystal_rule.cpp
#include "discover_crystal_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        DiscoverCrystalRule::DiscoverCrystalRule()
        {
            LOG_INFO("DiscoverCrystalRule::DiscoverCrystalRule()");
        }
        bool DiscoverCrystalRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("DiscoverCrystalRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("DiscoverCrystalRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void DiscoverCrystalRule::reset()
        {
            LOG_INFO_FMT("DiscoverCrystalRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json DiscoverCrystalRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("DiscoverCrystalRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus DiscoverCrystalRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("DiscoverCrystalRule::rule_logic()");
            int crystal_count = 0;
            std::vector<int> crystal_track_ids;
            for (const auto &box : packet->detections)
            {
                if (box.class_name != "crystal")
                { 
                    continue;
                } 
                bool in_zone = zone_points.empty() ? true : ZoneValidator::pointInPolygon(PixelPoint(box.x + box.w / 2, box.y + box.h / 2), zone_points);
                if (in_zone)
                {
                    crystal_count++;
                    crystal_track_ids.push_back(box.track_id);
                }
            }
            if (crystal_count < 1)
                return RuleStatus::RULE_STATUS_OK;
                
            updateZoneEvent(zone_no, packet, crystal_track_ids);

            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("discover_crystal", DiscoverCrystalRule)
    }
}
