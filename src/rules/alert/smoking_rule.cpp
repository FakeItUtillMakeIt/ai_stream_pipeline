// src/rules/alert/smoking_rule.cpp
#include "smoking_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        SmokingRule::SmokingRule()
        {
            LOG_INFO("SmokingRule::SmokingRule()");
        }

        bool SmokingRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("SmokingRule::initialize()");
            try
            {
                LOG_INFO_FMT("SmokingRule::initialize() config: {}", config.dump().c_str());
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
                LOG_WARN_FMT("SmokingRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("SmokingRule::initialize() success");

            // 判断区域是否有效/配置
            uint8_t invaild_zone_count = 0;

            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("SmokingRule::initialize() zone {} is invalid", det_zone.first);
                    invaild_zone_count++;
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }
            // 如果所有区域都无效，则全域监测(不进行区域过滤)
            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("SmokingRule::initialize() all zones are invalid, global monitoring");
            }

            return true;
        }

        RuleStatus SmokingRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult &alert_result,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("SmokingRule::process()");
            std::lock_guard<std::mutex> lock(mutex_);
            if (!packet)
                return RuleStatus::RULE_STATUS_FAIL;
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
                if (it->second.status == AlertStatus::ALERT_STATUS_END)
                {
                    it->second.status = AlertStatus::ALERT_STATUS_DEFAULT;
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

        void SmokingRule::reset()
        {
            LOG_INFO_FMT("SmokingRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json SmokingRule::getStatistics() const
        {
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus SmokingRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("SmokingRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            std::vector<core::InferenceResultPacket::BBox> head_boxes;
            std::vector<core::InferenceResultPacket::BBox> helmet_boxes;
            std::vector<core::InferenceResultPacket::BBox> smoking_boxes;
            int person_smoking_count = 0;
            std::vector<int> person_smoking_track_ids;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
                }
                else if (detection.class_name == "head")
                {
                    head_boxes.push_back(detection);
                }
                else if (detection.class_name == "helmet")
                {
                    helmet_boxes.push_back(detection);
                }
                else if (detection.class_name == "smoking")
                {
                    smoking_boxes.push_back(detection);
                }
            }
            // 人员是否抽烟
            if (person_boxes.empty() || smoking_boxes.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            for (const auto &person_box : person_boxes)
            {
                bool person_has_smoking = false;
                std::vector<PixelPoint> person_zone{PixelPoint(person_box.x, person_box.y), PixelPoint(person_box.x + person_box.w, person_box.y), PixelPoint(person_box.x + person_box.w, person_box.y + person_box.h), PixelPoint(person_box.x, person_box.y + person_box.h)};
                // 筛选当前人体范围内的人头
                std::vector<core::InferenceResultPacket::BBox> head_boxes_in_person_box;
                for (const auto &head_box : head_boxes)
                {
                    if (ZoneValidator::pointInPolygon(PixelPoint(head_box.x + head_box.w / 2, head_box.y + head_box.h / 2), person_zone))
                    {
                        head_boxes_in_person_box.push_back(head_box);
                    }
                }
                // 筛选当前人体范围内是否有头盔
                std::vector<core::InferenceResultPacket::BBox> helmet_boxes_in_person_box;
                for (const auto &helmet_box : helmet_boxes)
                {
                    if (ZoneValidator::pointInPolygon(PixelPoint(helmet_box.x + helmet_box.w / 2, helmet_box.y + helmet_box.h / 2), person_zone))
                    {
                        helmet_boxes_in_person_box.push_back(helmet_box);
                    }
                }
                // 合并头和头盔
                head_boxes_in_person_box.insert(head_boxes_in_person_box.end(), helmet_boxes_in_person_box.begin(), helmet_boxes_in_person_box.end());
                if (head_boxes_in_person_box.empty())
                    continue;
                // 检测抽烟框是否与人头/头盔相交
                for (const auto &head_box : head_boxes_in_person_box)
                {
                    std::vector<PixelPoint> head_zone{PixelPoint(head_box.x, head_box.y), PixelPoint(head_box.x + head_box.w, head_box.y), PixelPoint(head_box.x + head_box.w, head_box.y + head_box.h), PixelPoint(head_box.x, head_box.y + head_box.h)};
                    for (const auto &smoking_box : smoking_boxes)
                    {
                        std::vector<PixelPoint> smoking_zone{PixelPoint(smoking_box.x, smoking_box.y), PixelPoint(smoking_box.x + smoking_box.w, smoking_box.y), PixelPoint(smoking_box.x + smoking_box.w, smoking_box.y + smoking_box.h), PixelPoint(smoking_box.x, smoking_box.y + smoking_box.h)};
                        if (ZoneValidator::boxIsIntersect(head_zone, smoking_zone))
                        {
                            person_has_smoking = true;
                            break;
                        }
                    }
                    if (person_has_smoking)
                        break;
                }
                if (person_has_smoking)
                {
                    person_smoking_count++;
                    person_smoking_track_ids.push_back(person_box.track_id);
                    break;
                }
            }
            if (person_smoking_count <= 0)
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
                alert_target.object_ids = person_smoking_track_ids;
                zone_alert_map_.insert(std::make_pair(zone_no, alert_target));
            }
            else
            {
                auto &alert_target = it->second;
                alert_target.non_update_count = 0;
                alert_target.duration_ms = packet->timestamp_ms - alert_target.detect_ms;
                alert_target.object_ids = person_smoking_track_ids;
            }
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("smoking", SmokingRule)
    }
}