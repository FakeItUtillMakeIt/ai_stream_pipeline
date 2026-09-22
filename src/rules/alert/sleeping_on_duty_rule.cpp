// src/rules/alert/sleeping_on_duty_rule.cpp
#include "sleeping_on_duty_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        SleepingOnDutyRule::SleepingOnDutyRule() : station_detector_(), sleeping_on_duty_threshold_(50)
        {
            LOG_INFO("SleepingOnDutyRule::SleepingOnDutyRule()");
            station_detector_.setMinStayDuration(5000); // 设置最小停留时长为5秒
        }

        bool SleepingOnDutyRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("SleepingOnDutyRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("SleepingOnDutyRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void SleepingOnDutyRule::onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            // 使用岗位检测器更新岗位区域（每帧一次）
            all_station_regions_ = station_detector_.getAllStations(packet->detections);
        }

        void SleepingOnDutyRule::reset()
        {
            LOG_INFO_FMT("SleepingOnDutyRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json SleepingOnDutyRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus SleepingOnDutyRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("SleepingOnDutyRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> sleeping_boxes;
            int sleeping_on_duty_count = 0;
            std::vector<core::InferenceResultPacket::BBox> person_not_in_station_box;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "sleeping")
                {
                    sleeping_boxes.push_back(detection);
                }
            }

            if (all_station_regions_.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            // 根据生成的岗位区域，判断人员是否在岗位内
            for (const auto &station_region : all_station_regions_)
            {
                int station_id = station_region.track_id;
                bool sleeping_on_duty = false;
                // 检查当前区域是否有人
                for (const auto &sleeping_box : sleeping_boxes)
                {
                    if (ZoneValidator::pointInPolygon(PixelPoint(sleeping_box.x + sleeping_box.w / 2, sleeping_box.y + sleeping_box.h / 2), station_region.region))
                    {
                        sleeping_on_duty = true;
                        break;
                    }
                }
                // 如果岗位有人且在睡觉，更新睡岗计数器
                if (sleeping_on_duty)
                {
                    if (sleeping_on_duty_counter_map_.find(station_id) == sleeping_on_duty_counter_map_.end())
                    {
                        sleeping_on_duty_counter_map_[station_id] = 1;
                    }
                    else
                    {
                        sleeping_on_duty_counter_map_[station_id]++;
                    }
                    // 判断是否达到睡岗阈值
                    if (sleeping_on_duty_counter_map_[station_id] >= sleeping_on_duty_threshold_)
                    {
                        core::InferenceResultPacket::BBox absent_box;
                        absent_box.x = station_region.region[0].x;
                        absent_box.y = station_region.region[0].y;
                        absent_box.w = station_region.region[1].x - station_region.region[0].x;
                        absent_box.h = station_region.region[2].y - station_region.region[0].y;
                        person_not_in_station_box.push_back(absent_box);
                        sleeping_on_duty_count++;
                    }
                }
                else
                {
                    // 人员清醒，重置睡岗计数（否则计数器闩锁，防抖只生效一次）
                    sleeping_on_duty_counter_map_[station_id] = 0;
                }
            }
            if (sleeping_on_duty_count <= 0)
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            updateZoneEvent(zone_no, packet, {});

            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("sleep_on_duty", SleepingOnDutyRule)
    }
}