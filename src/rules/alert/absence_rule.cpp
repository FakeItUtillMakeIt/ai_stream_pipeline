// src/rules/alert/absence_rule.cpp
#include "absence_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        AbsenceRule::AbsenceRule() : station_detector_(),absent_threshold_(50)
        {
            LOG_INFO("AbsenceRule::AbsenceRule()");
            station_detector_.setMinStayDuration(5000); // 设置最小停留时长为5秒
        }

        bool AbsenceRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("AbsenceRule::initialize()");
            try
            {
                LOG_INFO_FMT("AbsenceRule::initialize() config: {}", config.dump().c_str());
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name",""));
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
                LOG_WARN_FMT("AbsenceRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("AbsenceRule::initialize() success");

            // 判断区域是否有效/配置
            uint8_t invaild_zone_count = 0;

            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("AbsenceRule::process() zone {} is invalid", det_zone.first);
                    invaild_zone_count++;
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }
            // 如果所有区域都无效，则全域监测(不进行区域过滤)
            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("AbsenceRule::process() all zones are invalid, global monitoring");
            }

            return true;
        }

        RuleStatus AbsenceRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult &alert_result,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("AbsenceRule::process()");
            std::lock_guard<std::mutex> lock(mutex_);
            if (!packet)
                return RuleStatus::RULE_STATUS_FAIL;
            // 使用岗位检测器更新岗位区域
            all_station_regions_ = station_detector_.getAllStations(packet->detections);
            if (all_station_regions_.empty())
            {
                LOG_INFO_FMT("AbsenceRule::process() current no station detected");
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
                    it->second.description = getName() + " Lasting";
                }
                if (it->second.duration_ms > alert_duration_ms_ && it->second.status == AlertStatus::ALERT_STATUS_DEFAULT)
                {
                    // 生成一个告警id
                    it->second.status = AlertStatus::ALERT_STATUS_OCCUR;
                    it->second.alert_name = getName();
                    it->second.alert_type = getType();
                    it->second.description = getName() + " Occur";
                }
                if (it->second.non_update_count == max_disappear_count_)
                {
                    it->second.status = AlertStatus::ALERT_STATUS_END;
                    it->second.description = getName() + " End";
                }
                it++;
            }
            return RuleStatus::RULE_STATUS_OK;
        }

        void AbsenceRule::reset()
        {
            LOG_INFO_FMT("AbsenceRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json AbsenceRule::getStatistics() const
        {
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus AbsenceRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("AbsenceRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            int person_not_in_station_count = 0;
            std::vector<core::InferenceResultPacket::BBox> person_not_in_station_box;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
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
                bool station_occupied = false;
                // 检查当前区域是否有人
                for (const auto &person_box : person_boxes)
                {
                    if (ZoneValidator::pointInPolygon(PixelPoint(person_box.x + person_box.w / 2, person_box.y + person_box.h / 2), station_region.region))
                    {
                        station_occupied = true;
                        break;
                    }
                }
                // 如果岗位无人且之前有人，更新离岗计数器
                if (!station_occupied)
                {
                    if (m_absent_counter_map_.find(station_id) == m_absent_counter_map_.end())
                    {
                        m_absent_counter_map_[station_id] = 1;
                    }
                    else
                    {
                        m_absent_counter_map_[station_id]++;
                    }
                    // 判断是否达到离岗阈值
                    if (m_absent_counter_map_[station_id] >= absent_threshold_)
                    {
                        core::InferenceResultPacket::BBox absent_box;
                        absent_box.x = station_region.region[0].x;
                        absent_box.y = station_region.region[0].y;
                        absent_box.w = station_region.region[1].x - station_region.region[0].x;
                        absent_box.h = station_region.region[2].y - station_region.region[0].y;
                        person_not_in_station_box.push_back(absent_box);
                        person_not_in_station_count++;
                    }
                }
            }
            if (person_not_in_station_count <= 0)
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
        REGISTER_ALERT_RULE("absence", AbsenceRule)
    }
}