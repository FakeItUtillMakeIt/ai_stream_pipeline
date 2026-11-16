// src/rules/alert/phone_call_rule.cpp
#include "phone_call_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        PhoneCallRule::PhoneCallRule()
        {
            LOG_INFO("PhoneCallRule::PhoneCallRule()");
        }

        bool PhoneCallRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("PhoneCallRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("PhoneCallRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void PhoneCallRule::reset()
        {
            LOG_INFO_FMT("PhoneCallRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json PhoneCallRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus PhoneCallRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("PhoneCallRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            std::vector<core::InferenceResultPacket::BBox> head_boxes;
            std::vector<core::InferenceResultPacket::BBox> helmet_boxes;
            std::vector<core::InferenceResultPacket::BBox> phone_boxes;
            int person_phone_call_count = 0;
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
            for (const auto &person_box : person_boxes)
            {
                // P0：只判定落在本 zone 内的人员
                if (!boxInZone(person_box, zone_points))
                    continue;
                bool person_phone_call = false;

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
                // P1：电话框须与头部有实质重叠（贴耳/贴脸），而不是“任意接触”。
                for (const auto &phone_box : phone_boxes)
                {
                    bool phone_in_head = false;
                    for (const auto &head_box : person_head_boxes)
                    {
                        if (centerInsideBox(phone_box, head_box) ||
                            overlapRatioInParent(phone_box, head_box) >= 0.25f)
                        {
                            phone_in_head = true;
                            break;
                        }
                    }
                    if (phone_in_head)
                    {
                        person_phone_call = true;
                        person_phone_call_count++;
                        person_phone_call_track_ids.push_back(person_box.track_id);
                        break;
                    }
                }
            }
            if (person_phone_call_count <= 0)
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            updateZoneEvent(zone_no, packet, person_phone_call_track_ids);
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("phone_call", PhoneCallRule)
    }
}