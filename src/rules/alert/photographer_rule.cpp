// src/rules/alert/photographer_rule.cpp
#include "photographer_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        PhotographerRule::PhotographerRule() 
        {
            LOG_INFO("PhotographerRule::PhotographerRule()");
        }

        bool PhotographerRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("PhotographerRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("PhotographerRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void PhotographerRule::reset()
        {
            LOG_INFO_FMT("PhotographerRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json PhotographerRule::getStatistics() const
        {
            LOG_INFO_FMT("PhotographerRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus PhotographerRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("PhotographerRule::rule_logic()");
            std::vector<ai_stream::core::InferenceResultPacket::BBox> person_boxes;
            std::vector<ai_stream::core::InferenceResultPacket::BBox> camera_boxes;
            std::vector<ai_stream::core::InferenceResultPacket::BBox> pad_boxes;
            std::vector<int> person_track_ids;
            for (const auto &detection : packet->detections)
            {
                bool in_zone = zone_points.empty() ? true : ZoneValidator::pointInPolygon(PixelPoint(detection.x + detection.w / 2, detection.y + detection.h / 2), zone_points);
                if (!in_zone)
                {
                    continue;
                }
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
                }
                if (detection.class_name == "camera")
                {
                    camera_boxes.push_back(detection);
                }
                if (detection.class_name == "pad")
                {
                    pad_boxes.push_back(detection);
                }
            }

            if (person_boxes.empty() || camera_boxes.empty() || pad_boxes.empty())
                return RuleStatus::RULE_STATUS_OK;
            // todo: 完善揽拍逻辑
            bool is_photographer = false;
            for (const auto &person_box : person_boxes)
            { 
                // 查看pad框和camera框是否都与同一个person框相交
                bool is_person_camera_intersect = false;
                bool is_person_pad_intersect = false;
                for (const auto &camera_box : camera_boxes)
                {
                    if (ZoneValidator::boxIsIntersect(
                        std::vector<PixelPoint>{PixelPoint(person_box.x, person_box.y), PixelPoint(person_box.x + person_box.w, person_box.y), PixelPoint(person_box.x + person_box.w, person_box.y + person_box.h), PixelPoint(person_box.x, person_box.y + person_box.h)},
                        std::vector<PixelPoint>{PixelPoint(camera_box.x, camera_box.y), PixelPoint(camera_box.x + camera_box.w, camera_box.y), PixelPoint(camera_box.x + camera_box.w, camera_box.y + camera_box.h), PixelPoint(camera_box.x, camera_box.y + camera_box.h)}
                    ))
                    {
                        is_person_camera_intersect = true;
                    }
                }
                for (const auto &pad_box : pad_boxes)
                {
                    if (ZoneValidator::boxIsIntersect(
                        std::vector<PixelPoint>{PixelPoint(person_box.x, person_box.y), PixelPoint(person_box.x + person_box.w, person_box.y), PixelPoint(person_box.x + person_box.w, person_box.y + person_box.h), PixelPoint(person_box.x, person_box.y + person_box.h)},
                        std::vector<PixelPoint>{PixelPoint(pad_box.x, pad_box.y), PixelPoint(pad_box.x + pad_box.w, pad_box.y), PixelPoint(pad_box.x + pad_box.w, pad_box.y + pad_box.h), PixelPoint(pad_box.x, pad_box.y + pad_box.h)}
                    ))
                    {
                        is_person_pad_intersect = true;
                    }
                }
                if (is_person_camera_intersect && is_person_pad_intersect)
                {
                    person_track_ids.push_back(person_box.track_id);
                    is_photographer = true;
                }
            }
            if (!is_photographer)
            { 
                return RuleStatus::RULE_STATUS_OK;
            }
            updateZoneEvent(zone_no, packet, person_track_ids);
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("photographer", PhotographerRule)
    }
}
