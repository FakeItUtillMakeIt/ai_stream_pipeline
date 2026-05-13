// src/rules/alert/missing_helmet_rule.cpp
#include "missing_helmet_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        MissingHelmetRule::MissingHelmetRule()
        {
            LOG_INFO("MissingHelmetRule::MissingHelmetRule()");
        }

        bool MissingHelmetRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("MissingHelmetRule::initialize()");
            try
            {
                LOG_INFO_FMT("MissingHelmetRule::initialize() config: {}", config.dump().c_str());

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
                LOG_WARN_FMT("MissingHelmetRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("MissingHelmetRule::initialize() success");

            // 判断区域是否有效/配置
            uint8_t invaild_zone_count = 0;

            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("MissingHelmetRule::process() zone {} is invalid", det_zone.first);
                    invaild_zone_count++;
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }
            // 如果所有区域都无效，则全域监测(不进行区域过滤)
            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("MissingHelmetRule::process() all zones are invalid, global monitoring");
            }

            return true;
        }

        RuleStatus MissingHelmetRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult &alert_result,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("MissingHelmetRule::process()");
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

        void MissingHelmetRule::reset()
        {
            LOG_INFO_FMT("MissingHelmetRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json MissingHelmetRule::getStatistics() const
        {
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus MissingHelmetRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("MissingHelmetRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            std::vector<core::InferenceResultPacket::BBox> head_boxes;
            std::vector<core::InferenceResultPacket::BBox> helmet_boxes;
            int person_not_helmet_count = 0;
            std::vector<int> not_helmet_track_ids;
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
            }
            // 人员是否佩戴安全帽
            if (person_boxes.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            for (auto &person_box : person_boxes)
            {
                bool person_has_helmet = false;
                bool person_has_head = false;
                std::vector<PixelPoint> person_zone;
                person_zone.push_back(PixelPoint(person_box.x, person_box.y));
                person_zone.push_back(PixelPoint(person_box.x + person_box.w, person_box.y));
                person_zone.push_back(PixelPoint(person_box.x + person_box.w, person_box.y + person_box.h));
                person_zone.push_back(PixelPoint(person_box.x, person_box.y + person_box.h));
                for (auto &helmet_box : helmet_boxes)
                {
                    person_has_helmet |= ZoneValidator::pointInPolygon(
                        PixelPoint(helmet_box.x + helmet_box.w / 2, helmet_box.y + helmet_box.h / 2),
                        person_zone);
                    if (person_has_helmet)
                        break;
                }
                for (auto &head : head_boxes)
                {
                    person_has_head |= ZoneValidator::pointInPolygon(
                        PixelPoint(head.x + head.w / 2, head.y + head.h / 2),
                        person_zone);
                    if (person_has_head)
                        break;
                }
                // 有安全帽+有人头，检查安全帽是否在人头上
                if (person_has_helmet && person_has_head)
                {
                    bool properly_wearing_helmet = false;
                    for (auto &head : head_boxes)
                    {
                        if (!ZoneValidator::pointInPolygon(
                                PixelPoint(head.x + head.w / 2, head.y + head.h / 2),
                                person_zone))
                        {
                            continue;
                        }
                        for (auto &helmet : helmet_boxes)
                        {
                            if (!ZoneValidator::pointInPolygon(
                                    PixelPoint(helmet.x + helmet.w / 2, helmet.y + helmet.h / 2),
                                    person_zone))
                            {
                                continue;
                            }
                            // 安全帽中心在人头内，认为佩戴正确
                            if (ZoneValidator::pointInPolygon(
                                    PixelPoint(helmet.x + helmet.w / 2, helmet.y + helmet.h / 2),
                                    std::vector<PixelPoint>{PixelPoint(head.x, head.y), PixelPoint(head.x + head.w, head.y), PixelPoint(head.x + head.w, head.y + head.h), PixelPoint(head.x, head.y + head.h)}))
                            {
                                properly_wearing_helmet = true;
                                break;
                            }
                        }
                        if (properly_wearing_helmet)
                            break;
                    }
                    if (!properly_wearing_helmet)
                    {
                        person_not_helmet_count++;
                        not_helmet_track_ids.push_back(person_box.track_id);
                    }
                }
                // 无安全帽但有人头，认为未佩戴安全帽
                else if (!person_has_helmet && person_has_head)
                {
                    person_not_helmet_count++;
                    not_helmet_track_ids.push_back(person_box.track_id);
                }
                // 其他情况（无安全帽无头，或有安全帽但无头）不纳入统计
            }
            if (person_not_helmet_count <= 0)
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
                alert_target.object_ids = not_helmet_track_ids;
                zone_alert_map_.insert(std::make_pair(zone_no, alert_target));
            }
            else
            {
                auto &alert_target = it->second;
                alert_target.non_update_count = 0;
                alert_target.duration_ms = packet->timestamp_ms - alert_target.detect_ms;
                alert_target.object_ids = not_helmet_track_ids;
            }
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("missing_helmet", MissingHelmetRule)
    }
}