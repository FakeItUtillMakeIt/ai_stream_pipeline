// src/rules/alert/missing_work_clothes_rule.cpp
#include "missing_work_clothes_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        MissingWorkClothesRule::MissingWorkClothesRule()
        {
            LOG_INFO("MissingWorkClothesRule::MissingWorkClothesRule()");
        }

        bool MissingWorkClothesRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("MissingWorkClothesRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("MissingWorkClothesRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void MissingWorkClothesRule::reset()
        {
            LOG_INFO_FMT("MissingWorkClothesRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json MissingWorkClothesRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus MissingWorkClothesRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("MissingWorkClothesRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            std::vector<core::InferenceResultPacket::BBox> work_clothes_boxes;
            int person_not_work_clothes_count = 0;
            std::vector<int> not_work_clothes_track_ids;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
                }
                else if (detection.class_name.find("clothes") != std::string::npos || detection.class_name.find("uniform") != std::string::npos)
                {
                    work_clothes_boxes.push_back(detection);
                }
            }
            // 遍历每个person，判断是否穿了工作服
            for (const auto &person_box : person_boxes)
            {
                bool person_has_work_clothes = false;
                for (const auto &work_clothes_box : work_clothes_boxes)
                {
                    if (work_clothes_box.track_id == person_box.track_id)
                    {
                        person_has_work_clothes = true;
                        break;
                    }
                }
                if (!person_has_work_clothes)
                {
                    person_not_work_clothes_count++;
                    not_work_clothes_track_ids.push_back(person_box.track_id);
                }
            }
            if (person_not_work_clothes_count > 0)
            {
                auto it = zone_alert_map_.find(zone_no);
                if (it == zone_alert_map_.end())
                {
                    auto alert_target = AlertEvent();
                    alert_target.detect_ms = packet->timestamp_ms;
                    alert_target.zone_no = zone_no;
                    alert_target.non_update_count = 0;
                    alert_target.duration_ms = 0;
                    alert_target.object_ids = not_work_clothes_track_ids;
                    zone_alert_map_.insert(std::make_pair(zone_no, alert_target));
                }
                else
                {
                    auto &alert_target = it->second;
                    alert_target.non_update_count = 0;
                    alert_target.duration_ms = packet->timestamp_ms - alert_target.detect_ms;
                    alert_target.object_ids = not_work_clothes_track_ids;
                }
            }
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("missing_work_clothes", MissingWorkClothesRule)
    }
}