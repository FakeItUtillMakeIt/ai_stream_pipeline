// src/rules/alert/discover_smoking_rule.cpp
#include "discover_smoke_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        DiscoverSmokeRule::DiscoverSmokeRule() 
        {
            LOG_INFO("DiscoverSmokeRule::DiscoverSmokeRule()");
        }

        bool DiscoverSmokeRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("DiscoverSmokeRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("DiscoverSmokeRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void DiscoverSmokeRule::reset()
        {
            LOG_INFO_FMT("DiscoverSmokeRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json DiscoverSmokeRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("DiscoverSmokeRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus DiscoverSmokeRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("DiscoverSmokeRule::rule_logic()");
            int smoke_count = 0;
            std::vector<int> smoke_track_ids;
            for (const auto &box : packet->detections)
            {
                if (box.class_name != "smoke")
                { 
                    continue;
                } 
                bool in_zone = zone_points.empty() ? true : ZoneValidator::pointInPolygon(PixelPoint(box.x + box.w / 2, box.y + box.h / 2), zone_points);
                if (in_zone)
                {
                    smoke_count++;
                    smoke_track_ids.push_back(box.track_id);
                }
            }
            if (smoke_count < 1)
                return RuleStatus::RULE_STATUS_OK;
                
            updateZoneEvent(zone_no, packet, smoke_track_ids);

            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("discover_smoke", DiscoverSmokeRule)
    }
}
