// src/rules/alert/discover_hose_cutoff_rule.cpp
#include "discover_hose_cutoff_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        DiscoverHoseCutoffRule::DiscoverHoseCutoffRule() 
        {
            LOG_INFO("DiscoverHoseCutoffRule::DiscoverHoseCutoffRule()");
        }
        bool DiscoverHoseCutoffRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("DiscoverHoseCutoffRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("DiscoverHoseCutoffRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void DiscoverHoseCutoffRule::reset()
        {
            LOG_INFO_FMT("DiscoverHoseCutoffRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json DiscoverHoseCutoffRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("DiscoverHoseCutoffRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus DiscoverHoseCutoffRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("DiscoverHoseCutoffRule::rule_logic()");
            int hose_count = 0;
            std::vector<int> hose_track_ids;
            for (const auto &box : packet->detections)
            {
                if (box.class_name != "hose")
                { 
                    continue;
                } 
                bool in_zone = zone_points.empty() ? true : ZoneValidator::pointInPolygon(PixelPoint(box.x + box.w / 2, box.y + box.h / 2), zone_points);
                if (in_zone)
                {
                    hose_count++;
                    hose_track_ids.push_back(box.track_id);
                }
            }
            if (hose_count < 1)
                return RuleStatus::RULE_STATUS_OK;
                
            updateZoneEvent(zone_no, packet, hose_track_ids);

            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("discover_hose_cutoff", DiscoverHoseCutoffRule)
    }
}
