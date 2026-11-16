// src/rules/alert/missing_safety_belt_rule.cpp
#include "missing_safety_belt_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        MissingSafetyBeltRule::MissingSafetyBeltRule()
        {
            LOG_INFO("MissingSafetyBeltRule::MissingSafetyBeltRule()");
        }

        bool MissingSafetyBeltRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("MissingSafetyBeltRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
                if (config.contains("min_person_h") && config["min_person_h"].is_number_integer())
                {
                    min_person_h_ = config["min_person_h"].get<int>();
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("MissingSafetyBeltRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void MissingSafetyBeltRule::reset()
        {
            LOG_INFO_FMT("MissingSafetyBeltRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json MissingSafetyBeltRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus MissingSafetyBeltRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("MissingSafetyBeltRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            std::vector<core::InferenceResultPacket::BBox> head_boxes;
            std::vector<core::InferenceResultPacket::BBox> helmet_boxes;
            std::vector<core::InferenceResultPacket::BBox> safety_belt_boxes;
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
                else if (detection.class_name == "safety_belt")
                {
                    safety_belt_boxes.push_back(detection);
                }
            }

            int person_not_safety_belt_count = 0;
            std::vector<int> not_safety_belt_track_ids;
            for (const auto &person_box : person_boxes)
            {
                // P0：只判定落在本 zone 内的人员
                if (!boxInZone(person_box, zone_points))
                    continue;
                // P1：远处/抖动小人不判，避免细目标（安全带）漏检即误报
                if (person_box.h < min_person_h_)
                    continue;

                // P1：先确认这是“可见的人”（头/盔与人体同框）。躯干不可见时
                // 无法判断是否系带，按“未知”处理而不是判为未系。
                bool confirmed_person = false;
                for (const auto &head : head_boxes)
                {
                    if (centerInsideBox(head, person_box) ||
                        overlapRatioInParent(head, person_box) >= 0.5f)
                    {
                        confirmed_person = true;
                        break;
                    }
                }
                if (!confirmed_person)
                {
                    for (const auto &helmet : helmet_boxes)
                    {
                        if (centerInsideBox(helmet, person_box) ||
                            overlapRatioInParent(helmet, person_box) >= 0.5f)
                        {
                            confirmed_person = true;
                            break;
                        }
                    }
                }
                if (!confirmed_person)
                    continue;

                // P1：安全带框归属到该人（中心在框内或大部分落在框内）
                bool person_has_safety_belt = false;
                for (const auto &safety_belt_box : safety_belt_boxes)
                {
                    if (centerInsideBox(safety_belt_box, person_box) ||
                        overlapRatioInParent(safety_belt_box, person_box) >= 0.5f)
                    {
                        person_has_safety_belt = true;
                        break;
                    }
                }
                if (!person_has_safety_belt)
                {
                    person_not_safety_belt_count++;
                    not_safety_belt_track_ids.push_back(person_box.track_id);
                }
            }
            if (person_not_safety_belt_count > 0)
            {
                updateZoneEvent(zone_no, packet, not_safety_belt_track_ids);
            }
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("missing_safety_belt", MissingSafetyBeltRule)
    }
}