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
                if (config.contains("min_person_h") && config["min_person_h"].is_number_integer())
                {
                    min_person_h_ = config["min_person_h"].get<int>();
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
                // P0：只判定落在本 zone 内的人员
                if (!boxInZone(person_box, zone_points))
                    continue;
                // P1：远处/抖动小人不判，避免头部/头盔漏检即误报
                if (person_box.h < min_person_h_)
                    continue;

                // P1：用“中心在框内或大部分落在框内”替代原来严格的中心点判定，
                // 降低检测框抖动导致的佩戴误判。
                std::vector<core::InferenceResultPacket::BBox> person_helmets;
                std::vector<core::InferenceResultPacket::BBox> person_heads;
                for (auto &helmet_box : helmet_boxes)
                {
                    if (centerInsideBox(helmet_box, person_box) ||
                        overlapRatioInParent(helmet_box, person_box) >= 0.5f)
                    {
                        person_helmets.push_back(helmet_box);
                    }
                }
                for (auto &head_box : head_boxes)
                {
                    if (centerInsideBox(head_box, person_box) ||
                        overlapRatioInParent(head_box, person_box) >= 0.5f)
                    {
                        person_heads.push_back(head_box);
                    }
                }

                const bool person_has_helmet = !person_helmets.empty();
                const bool person_has_head = !person_heads.empty();

                // 只有头盔、没有头：多半是头盔遮住头部导致 head 漏检 → 视为已佩戴
                if (person_has_helmet && !person_has_head)
                    continue;
                // 既无头也无头盔：无法确认，不计入
                if (!person_has_helmet && !person_has_head)
                    continue;

                bool properly_wearing_helmet = false;
                for (auto &head : person_heads)
                {
                    for (auto &helmet : person_helmets)
                    {
                        // 头盔中心在头部框内，或头盔大部分落在头部框内
                        if (centerInsideBox(helmet, head) ||
                            overlapRatioInParent(helmet, head) >= 0.3f)
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