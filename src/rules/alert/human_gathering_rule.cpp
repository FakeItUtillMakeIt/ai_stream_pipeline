// src/rules/alert/human_gathering_rule.cpp
#include "human_gathering_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        HumanGatheringRule::HumanGatheringRule()
        {
            LOG_INFO("HumanGatheringRule::HumanGatheringRule()");
        }

        bool HumanGatheringRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("HumanGatheringRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
                if (config.contains("gathering_thresh") && config["gathering_thresh"].is_array())
                {
                    for (size_t i = 0; i < config["gathering_thresh"].size(); i++)
                    {
                        if (config["gathering_thresh"][i].is_number_integer())
                        {
                            gathering_thresh_map_[uint8_t(i + 1)] = config["gathering_thresh"][i].get<int>();
                        }
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("HumanGatheringRule::initialize() exception: {}", e.what());
                return false;
            }
            if (!parseZones(config))
            {
                return false;
            }
            // 全域回退：聚集阈值取所有已配置区域阈值的最小值（默认 2）
            if (valid_intrusion_zones_.empty())
            {
                int min_thresh = 2;
                bool any = false;
                for (const auto &kv : gathering_thresh_map_)
                {
                    min_thresh = any ? std::min(min_thresh, kv.second) : kv.second;
                    any = true;
                }
                gathering_thresh_map_[global_zone_no_] = min_thresh;
            }
            return true;
        }

        void HumanGatheringRule::reset()
        {
            LOG_INFO_FMT("HumanGatheringRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json HumanGatheringRule::getStatistics() const
        {
            LOG_INFO_FMT("HumanGatheringRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus HumanGatheringRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("HumanGatheringRule::rule_logic()");
            int person_count = 0;
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
                    person_count++;
                    person_track_ids.push_back(detection.track_id);
                }
            }
            // 更新zone_alert_map_（未配置阈值的区域默认 2 人）
            const int thresh = gathering_thresh_map_.count(zone_no) ? gathering_thresh_map_.at(zone_no) : 2;
            if (person_count < thresh)
                return RuleStatus::RULE_STATUS_OK;
            updateZoneEvent(zone_no, packet, person_track_ids);

            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("human_gathering", HumanGatheringRule)
    }
}
