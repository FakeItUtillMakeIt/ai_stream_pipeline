// src/rules/alert/climbing_rule.cpp
#include "climbing_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        ClimbingRule::ClimbingRule() : climbing_detector_(ClimbingDetector::Config())
        {
            LOG_INFO("ClimbingRule::ClimbingRule()");
            action_recognition_mode_ = ActionRecongnitionType::ACTION_RECOGNITION_POSE;
            alert_duration_ms_ = 100;
        }

        bool ClimbingRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("ClimbingRule::initialize()");
            try
            {
                LOG_INFO_FMT("ClimbingRule::initialize() config: {}", config.dump().c_str());
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
                if (config.contains("action_recongnition_mode") && config["action_recongnition_mode"].is_string())
                {
                    std::string mode = config.value("action_recongnition_mode", "");
                    if (mode == "model")
                    {
                        action_recognition_mode_ = ActionRecongnitionType::ACTION_RECOGNITION_MODEL;
                    }
                    else if (mode == "pose")
                    {
                        action_recognition_mode_ = ActionRecongnitionType::ACTION_RECOGNITION_POSE;
                    }
                
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("ClimbingRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("ClimbingRule::initialize() success");

            // 判断区域是否有效/配置
            uint8_t invaild_zone_count = 0;
            int min_gathering_thresh = std::numeric_limits<int>::max();
            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("ClimbingRule::initialize() zone {} is invalid", det_zone.first);
                    invaild_zone_count++;
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }
            // 如果所有区域都无效，则全域监测(不进行区域过滤)
            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("ClimbingRule::initialize() all zones are invalid, global monitoring");
            }

            return true;
        }

        RuleStatus ClimbingRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult &alert_result,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("ClimbingRule::process()");
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
            for (auto it = zone_alert_map_.begin(); it != zone_alert_map_.end(); it++)
            {
                if (it->second.status != AlertStatus::ALERT_STATUS_OCCUR && it->second.status != AlertStatus::ALERT_STATUS_LAST && it->second.status != AlertStatus::ALERT_STATUS_END)
                {
                    continue;
                }
                alert_result.alert_events.push_back(it->second);
                alert_result.alert_count++;
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
                    it->second.alert_item_type = getAlertItemType();
                }
                if (it->second.status != AlertStatus::ALERT_STATUS_DEFAULT && it->second.non_update_count == max_disappear_count_)
                {
                    it->second.status = AlertStatus::ALERT_STATUS_END;
                }
                it->second.description = getName() + alert_status_map[it->second.status];
                LOG_INFO_FMT("ClimbingRule::process() zone {} alert status: {}, non_update_count: {}, duration_ms: {}", int(it->first), alert_status_map[it->second.status], int(it->second.non_update_count), int(it->second.duration_ms));
                it++;
            }
            return RuleStatus::RULE_STATUS_OK;
        }

        void ClimbingRule::reset()
        {
            LOG_INFO_FMT("ClimbingRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json ClimbingRule::getStatistics() const
        {
            LOG_INFO_FMT("ClimbingRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus ClimbingRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("ClimbingRule::rule_logic()");
            std::vector<ai_stream::core::InferenceResultPacket::BBox> person_boxes;
            std::vector<int> person_track_ids;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
                }
            }
            if (person_boxes.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            bool is_climbing = false;
            if (action_recognition_mode_==ActionRecongnitionType::ACTION_RECOGNITION_MODEL)
            {
                LOG_INFO_FMT("ClimbingRule::rule_logic() using action recognition model:{}", packet->action_results.size());
                if (packet->action_results.empty())
                    return RuleStatus::RULE_STATUS_OK;
                for (const auto &action_result : packet->action_results)
                {
                    LOG_INFO_FMT("ClimbingRule::rule_logic() action_result: track_id={}, action_label={}, confidence={}", action_result.track_id, action_result.action_label, action_result.confidence);
                    if (action_result.action_label != alertTypeMap[AlertType::CLIMBING])
                        continue;
                    is_climbing = true;
                }
            }
            if (action_recognition_mode_ == ActionRecongnitionType::ACTION_RECOGNITION_POSE)
            {// 调用打架检测器
                auto climb_result = climbing_detector_.process(person_boxes);
                is_climbing = climb_result.is_climbing;
                person_track_ids = climb_result.active_track_ids;
            }
            
            if(is_climbing)
            {
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
            }
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("climbing", ClimbingRule)
        // 兼容历史拼写错误的旧配置名
        REGISTER_ALERT_RULE("clambing", ClimbingRule)
    }
}
