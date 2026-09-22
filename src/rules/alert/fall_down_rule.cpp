// src/rules/alert/fall_down_rule.cpp
#include "fall_down_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        FallDownRule::FallDownRule()
        {
            LOG_INFO("FallDownRule::FallDownRule()");
        }

        bool FallDownRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("FallDownRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("FallDownRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void FallDownRule::reset()
        {
            LOG_INFO_FMT("FallDownRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json FallDownRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus FallDownRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("FallDownRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            std::vector<core::InferenceResultPacket::BBox> fall_down_boxes;
            int person_fall_down_count = 0;
            std::vector<int> fall_down_track_ids;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
                }
                else if (detection.class_name == "fall_down")
                {
                    fall_down_boxes.push_back(detection);
                }
            }

            if (person_boxes.empty() && fall_down_boxes.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            // 跌倒框中心是否在人体框中
            for (const auto &fall_down_box : fall_down_boxes)
            {
                person_fall_down_count++;
                fall_down_track_ids.push_back(fall_down_box.track_id);
            }
            
            if (person_fall_down_count <= 0)
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            updateZoneEvent(zone_no, packet, fall_down_track_ids);
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("fall_down", FallDownRule)
    }
}