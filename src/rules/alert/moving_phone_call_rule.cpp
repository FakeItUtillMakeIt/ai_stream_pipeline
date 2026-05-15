// src/rules/alert/moving_phone_call_rule.cpp
#include "moving_phone_call_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        MovingPhoneCallRule::MovingPhoneCallRule()
        {
            LOG_INFO("MovingPhoneCallRule::MovingPhoneCallRule()");
        }

        bool MovingPhoneCallRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("MovingPhoneCallRule::initialize()");
            try
            {
                LOG_INFO_FMT("MovingPhoneCallRule::initialize() config: {}", config.dump().c_str());
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
                LOG_WARN_FMT("MovingPhoneCallRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("MovingPhoneCallRule::initialize() success");

            // 判断区域是否有效/配置
            uint8_t invaild_zone_count = 0;

            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("MovingPhoneCallRule::process() zone {} is invalid", det_zone.first);
                    invaild_zone_count++;
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }
            // 如果所有区域都无效，则全域监测(不进行区域过滤)
            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("MovingPhoneCallRule::process() all zones are invalid, global monitoring");
            }

            return true;
        }

        RuleStatus MovingPhoneCallRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult &alert_result,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("MovingPhoneCallRule::process()");
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

        void MovingPhoneCallRule::reset()
        {
            LOG_INFO_FMT("MovingPhoneCallRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json MovingPhoneCallRule::getStatistics() const
        {
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus MovingPhoneCallRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("MovingPhoneCallRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            std::vector<core::InferenceResultPacket::BBox> head_boxes;
            std::vector<core::InferenceResultPacket::BBox> helmet_boxes;
            std::vector<core::InferenceResultPacket::BBox> phone_boxes;
            int phone_count = 0;
            int moving_phonecall_count = 0;
            std::vector<int> person_phone_call_track_ids;
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
                else if (detection.class_name == "phone")
                {
                    phone_boxes.push_back(detection);
                }
            }
            std::unordered_set<int> active_track_ids;
            active_track_ids.reserve(person_boxes.size());
            for (const auto &person_box : person_boxes)
            {
                if(person_box.track_id > 0)active_track_ids.insert(person_box.track_id);
                std::vector<PixelPoint> person_zone{
                    PixelPoint(person_box.x, person_box.y),
                    PixelPoint(person_box.x + person_box.w, person_box.y),
                    PixelPoint(person_box.x + person_box.w, person_box.y + person_box.h),
                    PixelPoint(person_box.x, person_box.y + person_box.h)};
                // 筛选人体范围内的人头
                std::vector<core::InferenceResultPacket::BBox> person_head_boxes;
                std::vector<core::InferenceResultPacket::BBox> person_helmet_boxes;
                for (const auto &head_box : head_boxes)
                {
                    if (ZoneValidator::pointInPolygon(
                            PixelPoint(head_box.x + head_box.w / 2, head_box.y + head_box.h / 2),
                            person_zone))
                    {
                        person_head_boxes.push_back(head_box);
                    }
                }
                // 筛选人体范围内的安全帽
                for (const auto &helmet_box : helmet_boxes)
                {
                    if (ZoneValidator::pointInPolygon(
                            PixelPoint(helmet_box.x + helmet_box.w / 2, helmet_box.y + helmet_box.h / 2),
                            person_zone))
                    {
                        person_helmet_boxes.push_back(helmet_box);
                    }
                }
                // 合并人头和安全帽
                person_head_boxes.insert(person_head_boxes.end(), person_helmet_boxes.begin(), person_helmet_boxes.end());
                if (person_head_boxes.empty())
                {
                    continue;
                }
                // 筛选人体范围内的电话框
                std::vector<core::InferenceResultPacket::BBox> person_phone_boxes;
                for (const auto &phone_box : phone_boxes)
                {
                    if (ZoneValidator::pointInPolygon(
                            PixelPoint(phone_box.x + phone_box.w / 2, phone_box.y + phone_box.h / 2),
                            person_zone))
                    {
                        person_phone_boxes.push_back(phone_box);
                    }
                }
                if(person_phone_boxes.empty())
                {
                    continue;
                }

                // 检查电话框是否与人头/安全帽范围相交
                bool phone_in_head = false;
                for (const auto &phone_box : person_phone_boxes)
                {
                    for (const auto &head_box : person_head_boxes)
                    {
                        if (ZoneValidator::boxIsIntersect(
                                std::vector<PixelPoint>{PixelPoint(phone_box.x, phone_box.y), PixelPoint(phone_box.x + phone_box.w, phone_box.y), PixelPoint(phone_box.x + phone_box.w, phone_box.y + phone_box.h), PixelPoint(phone_box.x, phone_box.y + phone_box.h)},
                                std::vector<PixelPoint>{PixelPoint(head_box.x, head_box.y), PixelPoint(head_box.x + head_box.w, head_box.y), PixelPoint(head_box.x + head_box.w, head_box.y + head_box.h), PixelPoint(head_box.x, head_box.y + head_box.h)}))
                        {
                            phone_in_head = true;
                            break;
                        }
                    }
                    if (phone_in_head)break;
                }
                if(!phone_in_head)continue;
                //检测到打电话
                phone_count++;
                //判断是否在移动
                bool is_moving = false;
                if(person_box.track_id > 0)
                {
                    is_moving = moving_pc_detector_.update_track(person_box.track_id, {person_box.x,person_box.y,person_box.w,person_box.y}, packet->frame_id);
                    //跟踪长度辅助判断
                    if(person_box.track_age > 10 && is_moving)is_moving=true;
                }
                if(is_moving)
                {
                    moving_phonecall_count++;
                    person_phone_call_track_ids.push_back(person_box.track_id);
                }
            }
            //清理不活跃的轨迹
            moving_pc_detector_.cleanup_old_tracks(active_track_ids);
            //更新map
            if (moving_phonecall_count <= 0)
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
                alert_target.object_ids = person_phone_call_track_ids;
                zone_alert_map_.insert(std::make_pair(zone_no, alert_target));
            }
            else
            {
                auto &alert_target = it->second;
                alert_target.non_update_count = 0;
                alert_target.duration_ms = packet->timestamp_ms - alert_target.detect_ms;
                alert_target.object_ids = person_phone_call_track_ids;
            }
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("moving_phone_call", MovingPhoneCallRule)
    }
}