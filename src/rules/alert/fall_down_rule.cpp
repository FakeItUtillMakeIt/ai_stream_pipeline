// src/rules/alert/fall_down_rule.cpp
#include "fall_down_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        FallDownRule::FallDownRule()
        {
            LOG_INFO("FallDownRule::FallDownRule()");
        }

        bool FallDownRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("FallDownRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("FallDownRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void FallDownRule::reset()
        {
            LOG_INFO_FMT("FallDownRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json FallDownRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus FallDownRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("FallDownRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> fall_down_boxes;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "fall_down")
                {
                    fall_down_boxes.push_back(detection);
                }
            }

            if (fall_down_boxes.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }

            // fall_down 与 person 是互斥类别：人跌倒时模型输出 fall_down 而非 person
            // （见 FallingRule 的 person -> down 轨迹类别转移），因此不能要求同帧存在
            // 重叠的 person 框，否则跌倒目标会被全部丢弃。误报控制交给基类的
            // alert_duration_ms 持续确认 + zone 过滤（可用 min_conf 调优）。
            std::vector<int> fall_down_track_ids;
            for (const auto &fall_down_box : fall_down_boxes)
            {
                // P0：只统计落在本 zone 内的跌倒目标
                if (!boxInZone(fall_down_box, zone_points))
                    continue;
                fall_down_track_ids.push_back(fall_down_box.track_id);
            }

            if (fall_down_track_ids.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            updateZoneEvent(zone_no, packet, fall_down_track_ids);
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("fall_down", FallDownRule)
    }
}