// src/rules/alert/fighting_rule.cpp
#include "fighting_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <algorithm>
#include <vector>

namespace ai_stream
{
    namespace rules
    {

        FightingRule::FightingRule() : fighting_detector_(FightingDetector::Config())
        {
            LOG_INFO("FightingRule::FightingRule()");
            action_recognition_mode_ = ActionRecongnitionType::ACTION_RECOGNITION_POSE;
        }

        bool FightingRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("FightingRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("FightingRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void FightingRule::onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            last_is_fighting_ = false;
            last_fight_track_ids_.clear();
            std::vector<ai_stream::core::InferenceResultPacket::BBox> person_boxes;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                    person_boxes.push_back(detection);
            }
            if (person_boxes.empty())
            {
                return;
            }
            if (action_recognition_mode_ == ActionRecongnitionType::ACTION_RECOGNITION_MODEL)
            {
                for (const auto &action_result : packet->action_results)
                {
                    if (action_result.action_label == alertTypeMap[AlertType::FIGHTING])
                        last_is_fighting_ = true;
                }
            }
            else if (action_recognition_mode_ == ActionRecongnitionType::ACTION_RECOGNITION_POSE)
            {
                auto fight_result = fighting_detector_.process(person_boxes);
                last_is_fighting_ = fight_result.is_fighting;
                last_fight_track_ids_ = fight_result.active_track_ids;
            }
        }

        void FightingRule::reset()
        {
            LOG_INFO_FMT("FightingRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json FightingRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("FightingRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus FightingRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            // 统计 zone 内人数（保持“至少2人”语义）；检测结果已在本帧 process() 计算一次
            int person_count = 0;
            std::vector<int> in_zone_track_ids;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name != "person")
                    continue;
                bool in_zone = zone_points.empty() ? true : ZoneValidator::pointInPolygon(PixelPoint(detection.x + detection.w / 2, detection.y + detection.h / 2), zone_points);
                if (in_zone)
                {
                    person_count++;
                    if (detection.track_id > 0)
                        in_zone_track_ids.push_back(detection.track_id);
                }
            }
            if (person_count < 2 || !last_is_fighting_)
                return RuleStatus::RULE_STATUS_OK;

            std::vector<int> fight_ids;
            if (action_recognition_mode_ == ActionRecongnitionType::ACTION_RECOGNITION_POSE)
            {
                if (last_fight_track_ids_.empty())
                {
                    fight_ids = in_zone_track_ids;
                }
                else
                {
                    for (int id : last_fight_track_ids_)
                    {
                        if (std::find(in_zone_track_ids.begin(), in_zone_track_ids.end(), id) != in_zone_track_ids.end())
                            fight_ids.push_back(id);
                    }
                    if (fight_ids.size() < 2)
                        return RuleStatus::RULE_STATUS_OK;
                }
            }

            updateZoneEvent(zone_no, packet, fight_ids);
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("fighting", FightingRule)
    }
}
