// src/rules/alert/sleeping_on_duty_rule.cpp
#include "sleeping_on_duty_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        SleepingOnDutyRule::SleepingOnDutyRule() : station_detector_(), sleeping_on_duty_threshold_(50)
        {
            LOG_INFO("SleepingOnDutyRule::SleepingOnDutyRule()");
            station_detector_.setMinStayDuration(5000); // 设置最小停留时长为5秒
        }

        bool SleepingOnDutyRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("SleepingOnDutyRule::initialize()");
            try
            {
                LOG_INFO_FMT("SleepingOnDutyRule::initialize() config: {}", config.dump().c_str());
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
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
                LOG_WARN_FMT("SleepingOnDutyRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("SleepingOnDutyRule::initialize() success");

            // 判断区域是否有效/配置
            uint8_t invaild_zone_count = 0;

            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("SleepingOnDutyRule::initialize() zone {} is invalid", det_zone.first);
                    invaild_zone_count++;
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }
            // 如果所有区域都无效，则全域监测(不进行区域过滤)
            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("SleepingOnDutyRule::initialize() all zones are invalid, global monitoring");
            }

            return true;
        }

        RuleStatus SleepingOnDutyRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult &alert_result,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("SleepingOnDutyRule::process()");
            std::lock_guard<std::mutex> lock(mutex_);
            if (!packet)
                return RuleStatus::RULE_STATUS_FAIL;
            // 使用岗位检测器更新岗位区域
            all_station_regions_ = station_detector_.getAllStations(packet->detections);
            if (all_station_regions_.empty())
            {
                LOG_INFO_FMT("SleepingOnDutyRule::process() current no station detected");
                return RuleStatus::RULE_STATUS_OK;
            }
            if (valid_intrusion_zones_.empty())
            {
                rule_logic(packet, global_zone_no_, {});
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
                    it->second.alert_name = getName();
                    it->second.alert_type = getType();
                }
                if (it->second.status != AlertStatus::ALERT_STATUS_DEFAULT && it->second.non_update_count == max_disappear_count_)
                {
                    it->second.status = AlertStatus::ALERT_STATUS_END;
                }
                it->second.description = getName() + alert_status_map[it->second.status];
                it++;
            }
            return RuleStatus::RULE_STATUS_OK;
        }

        void SleepingOnDutyRule::reset()
        {
            LOG_INFO_FMT("SleepingOnDutyRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json SleepingOnDutyRule::getStatistics() const
        {
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus SleepingOnDutyRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("SleepingOnDutyRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> sleeping_boxes;
            int sleeping_on_duty_count = 0;
            std::vector<core::InferenceResultPacket::BBox> person_not_in_station_box;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "sleeping")
                {
                    sleeping_boxes.push_back(detection);
                }
            }

            if (all_station_regions_.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            // 根据生成的岗位区域，判断人员是否在岗位内
            for (const auto &station_region : all_station_regions_)
            {
                int station_id = station_region.track_id;
                bool sleeping_on_duty = false;
                // 检查当前区域是否有人
                for (const auto &sleeping_box : sleeping_boxes)
                {
                    if (ZoneValidator::pointInPolygon(PixelPoint(sleeping_box.x + sleeping_box.w / 2, sleeping_box.y + sleeping_box.h / 2), station_region.region))
                    {
                        sleeping_on_duty = true;
                        break;
                    }
                }
                // 如果岗位有人且在睡觉，更新睡岗计数器
                if (sleeping_on_duty)
                {
                    if (sleeping_on_duty_counter_map_.find(station_id) == sleeping_on_duty_counter_map_.end())
                    {
                        sleeping_on_duty_counter_map_[station_id] = 1;
                    }
                    else
                    {
                        sleeping_on_duty_counter_map_[station_id]++;
                    }
                    // 判断是否达到睡岗阈值
                    if (sleeping_on_duty_counter_map_[station_id] >= sleeping_on_duty_threshold_)
                    {
                        core::InferenceResultPacket::BBox absent_box;
                        absent_box.x = station_region.region[0].x;
                        absent_box.y = station_region.region[0].y;
                        absent_box.w = station_region.region[1].x - station_region.region[0].x;
                        absent_box.h = station_region.region[2].y - station_region.region[0].y;
                        person_not_in_station_box.push_back(absent_box);
                        sleeping_on_duty_count++;
                    }
                }
            }
            if (sleeping_on_duty_count <= 0)
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            auto it = zone_alert_map_.find(zone_no);
            if (it == zone_alert_map_.end())
            {
                auto alert_target = AlertEvent();
                alert_target.detect_ms = packet->timestamp_ms;
                alert_target.zone_no = zone_no;
                alert_target.non_update_count = 0;
                alert_target.duration_ms = 0;

                zone_alert_map_.insert(std::make_pair(zone_no, alert_target));
            }
            else
            {
                auto &alert_target = it->second;
                alert_target.non_update_count = 0;
                alert_target.duration_ms = packet->timestamp_ms - alert_target.detect_ms;
            }

            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("absence", SleepingOnDutyRule)
    }
}