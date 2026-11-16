// src/rules/alert/sleeping_on_duty_rule.cpp
#include "sleeping_on_duty_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        SleepingOnDutyRule::SleepingOnDutyRule() : station_detector_(), sleeping_duration_ms_(2000)
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
                if (config.contains("sleeping_duration_ms") &&
                    config["sleeping_duration_ms"].is_number_integer())
                {
                    sleeping_duration_ms_ = config["sleeping_duration_ms"].get<int64_t>();
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
            // 无岗位帧不推进告警状态机（与原 process 早返回语义一致）
            skip_finalize_ = all_station_regions_.empty();
        }

        void SleepingOnDutyRule::reset()
        {
            LOG_INFO_FMT("SleepingOnDutyRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
            m_sleeping_start_ms_.clear();
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
            const int64_t now = packet->timestamp_ms;
            int sleeping_on_duty_count = 0;
            // 根据生成的岗位区域，判断人员是否在岗位内
            for (const auto &station_region : all_station_regions_)
            {
                // P0：只处理落在本 zone 内的岗位区域
                if (!polygonInZone(station_region.region, zone_points))
                    continue;
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
                // P1：时间基准判定，与帧率无关
                if (sleeping_on_duty)
                {
                    auto it = m_sleeping_start_ms_.find(station_id);
                    if (it == m_sleeping_start_ms_.end())
                    {
                        m_sleeping_start_ms_[station_id] = now;
                    }
                    else if (now - it->second >= sleeping_duration_ms_)
                    {
                        sleeping_on_duty_count++;
                    }
                }
                else
                {
                    // 人员清醒，清除睡岗计时
                    m_sleeping_start_ms_.erase(station_id);
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