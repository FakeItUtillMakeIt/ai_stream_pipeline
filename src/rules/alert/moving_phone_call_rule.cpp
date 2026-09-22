// src/rules/alert/moving_phone_call_rule.cpp
#include "moving_phone_call_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <unordered_set>
#include <vector>

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
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("MovingPhoneCallRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void MovingPhoneCallRule::onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            // 每帧只运行一次检测器（多 zone 时不得重复推进状态机），缓存移动打电话目标
            last_moving_phonecall_track_ids_.clear();
            {
                std::vector<core::InferenceResultPacket::BBox> person_boxes;
                std::vector<core::InferenceResultPacket::BBox> head_boxes;
                std::vector<core::InferenceResultPacket::BBox> helmet_boxes;
                std::vector<core::InferenceResultPacket::BBox> phone_boxes;
                for (const auto &detection : packet->detections)
                {
                    if (detection.class_name == "person")
                        person_boxes.push_back(detection);
                    else if (detection.class_name == "head")
                        head_boxes.push_back(detection);
                    else if (detection.class_name == "helmet")
                        helmet_boxes.push_back(detection);
                    else if (detection.class_name == "phone")
                        phone_boxes.push_back(detection);
                }
                std::unordered_set<int> active_track_ids;
                active_track_ids.reserve(person_boxes.size());
                for (const auto &person_box : person_boxes)
                {
                    if (person_box.track_id > 0)
                        active_track_ids.insert(person_box.track_id);
                    std::vector<PixelPoint> person_zone{
                        PixelPoint(person_box.x, person_box.y),
                        PixelPoint(person_box.x + person_box.w, person_box.y),
                        PixelPoint(person_box.x + person_box.w, person_box.y + person_box.h),
                        PixelPoint(person_box.x, person_box.y + person_box.h)};
                    std::vector<core::InferenceResultPacket::BBox> person_head_boxes;
                    std::vector<core::InferenceResultPacket::BBox> person_helmet_boxes;
                    for (const auto &head_box : head_boxes)
                    {
                        if (ZoneValidator::pointInPolygon(
                                PixelPoint(head_box.x + head_box.w / 2, head_box.y + head_box.h / 2),
                                person_zone))
                            person_head_boxes.push_back(head_box);
                    }
                    for (const auto &helmet_box : helmet_boxes)
                    {
                        if (ZoneValidator::pointInPolygon(
                                PixelPoint(helmet_box.x + helmet_box.w / 2, helmet_box.y + helmet_box.h / 2),
                                person_zone))
                            person_helmet_boxes.push_back(helmet_box);
                    }
                    person_head_boxes.insert(person_head_boxes.end(), person_helmet_boxes.begin(), person_helmet_boxes.end());
                    if (person_head_boxes.empty())
                        continue;
                    std::vector<core::InferenceResultPacket::BBox> person_phone_boxes;
                    for (const auto &phone_box : phone_boxes)
                    {
                        if (ZoneValidator::pointInPolygon(
                                PixelPoint(phone_box.x + phone_box.w / 2, phone_box.y + phone_box.h / 2),
                                person_zone))
                            person_phone_boxes.push_back(phone_box);
                    }
                    if (person_phone_boxes.empty())
                        continue;
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
                        if (phone_in_head)
                            break;
                    }
                    if (!phone_in_head)
                        continue;
                    bool is_moving = false;
                    if (person_box.track_id > 0)
                    {
                        is_moving = moving_pc_detector_.update_track(person_box.track_id, {person_box.x, person_box.y, person_box.x + person_box.w, person_box.y + person_box.h}, packet->frame_id);
                        if (person_box.track_age > 10 && is_moving)
                            is_moving = true;
                    }
                    if (is_moving)
                        last_moving_phonecall_track_ids_.push_back(person_box.track_id);
                }
                moving_pc_detector_.cleanup_old_tracks(active_track_ids);
            }
        }

        void MovingPhoneCallRule::reset()
        {
            LOG_INFO_FMT("MovingPhoneCallRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json MovingPhoneCallRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus MovingPhoneCallRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            // 检测结果已在本帧 process() 中计算一次，这里只做 zone 归属聚合
            if (last_moving_phonecall_track_ids_.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            updateZoneEvent(zone_no, packet, last_moving_phonecall_track_ids_);
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("moving_phone_call", MovingPhoneCallRule)
    }
}