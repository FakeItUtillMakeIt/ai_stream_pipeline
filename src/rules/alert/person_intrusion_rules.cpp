// src/rules/alert/person_instrusion_rules.cpp
#include "person_intrusion_rules.h"
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
                LOG_INFO_FMT("PersonIntrusionRule::initialize() config: {}", config.dump().c_str());

                if (config.contains("rule_zones") && config["rule_zones"].is_array())
                {
                    for (size_t i = 0; i < config["rule_zones"].size(); i++)
                    {
                        for (size_t k = 0; k < config["rule_zones"][i].size(); k++)
                        {
                            LOG_INFO_FMT("Rule zone {} add point {}: [{}, {}]", int(i + 1), int(k + 1), config["rule_zones"][i][k][0].get<float>(), config["rule_zones"][i][k][1].get<float>());
                            intrusion_zones_[uint8_t(i + 1)].push_back(PixelPoint(config["rule_zones"][i][k][0].get<float>(), config["rule_zones"][i][k][1].get<float>()));
                        }
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("PersonIntrusionRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("PersonIntrusionRule::initialize() success");

            // 判断区域是否有效/配置
            uint8_t invaild_zone_count = 0;

            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("PersonIntrusionRule::process() zone {} is invalid", det_zone.first);
                    invaild_zone_count++;
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }
            // 如果所有区域都无效，则全域监测(不进行区域过滤)
            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("PersonIntrusionRule::process() all zones are invalid, global monitoring");
            }

            return true;
        }

        RuleStatus PersonIntrusionRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult &alert_result,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("PersonIntrusionRule::process()");
            std::lock_guard<std::mutex> lock(mutex_);
            if (!packet)
                return RuleStatus::RULE_STATUS_FAIL;
            if (valid_intrusion_zones_.empty())
            {
                rule_logic(packet, 0, {});
            }
            else
            {
                for (const auto &zone : valid_intrusion_zones_)
                {
                    rule_logic(packet, zone.first, zone.second);
                }
            }
            // 更新告警结果
            uint8_t alert_count = alert_result.alert_events.size();
            for (auto it = zone_alert_map_.begin(); it != zone_alert_map_.end(); it++)
            {
                if (it->second.status != AlertStatus::ALERT_STATUS_OCCUR && it->second.status != AlertStatus::ALERT_STATUS_LAST && it->second.status != AlertStatus::ALERT_STATUS_END)
                {
                    continue;
                }
                alert_result.alert_events.push_back(it->second);
                alert_result.alert_count++;
                alert_count++;
            }

            // 更新map
            for (auto it = zone_alert_map_.begin(); it != zone_alert_map_.end();)
            {
                it->second.non_update_count++;
                if (it->second.non_update_count > max_disappear_count_)
                {
                    it = zone_alert_map_.erase(it);
                    continue;
                }
                if (it->second.status == AlertStatus::ALERT_STATUS_OCCUR)
                {
                    it->second.status = AlertStatus::ALERT_STATUS_LAST;
                }
                if (it->second.duration_ms > alert_duration_ms_ && it->second.status == AlertStatus::ALERT_STATUS_DEFAULT)
                {
                    // 生成一个告警id
                    it->second.status = AlertStatus::ALERT_STATUS_OCCUR;
                }
                if (it->second.non_update_count == max_disappear_count_)
                {
                    it->second.status = AlertStatus::ALERT_STATUS_END;
                }
                it++;
            }
            return RuleStatus::RULE_STATUS_OK;
        }

        void PersonIntrusionRule::reset()
        {
            LOG_INFO_FMT("PersonIntrusionRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json PersonIntrusionRule::getStatistics() const
        {
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
            // 更新zone_alert_map_
            if (!person_in_zone)
                return RuleStatus::RULE_STATUS_OK;
            auto it = zone_alert_map_.find(zone_no);
            if (it == zone_alert_map_.end())
            {
                auto alert_target = AlertEvent();
                alert_target.detect_ms = packet->timestamp_ms;
                alert_target.zone_no = zone_no;
                alert_target.non_update_count = 0;
                alert_target.duration_ms = 0;
                alert_target.object_ids = person_track_ids;
                zone_alert_map_.insert(std::make_pair(zone_no, alert_target));
            }
            else
            {
                auto &alert_target = it->second;
                alert_target.non_update_count = 0;
                alert_target.duration_ms = packet->timestamp_ms - alert_target.detect_ms;
                alert_target.object_ids = person_track_ids;
            }

            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("person_intrusion", PersonIntrusionRule)
    }
}
