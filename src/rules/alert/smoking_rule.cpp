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
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("SmokingRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
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
            updateZoneEvent(zone_no, packet, person_smoking_track_ids);
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("smoking", SmokingRule)
    }
}