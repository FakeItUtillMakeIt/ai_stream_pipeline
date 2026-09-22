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
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("MissingHelmetRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void MissingHelmetRule::reset()
        {
            LOG_INFO_FMT("MissingHelmetRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json MissingHelmetRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
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
            updateZoneEvent(zone_no, packet, not_helmet_track_ids);
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("missing_helmet", MissingHelmetRule)
    }
}