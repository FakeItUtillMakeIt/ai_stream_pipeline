// src/rules/alert/absence_rule.cpp
#include "absence_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        AbsenceRule::AbsenceRule() : station_detector_(), absent_threshold_(50)
        {
            LOG_INFO("AbsenceRule::AbsenceRule()");
            station_detector_.setMinStayDuration(5000); // 设置最小停留时长为5秒
        }

        bool AbsenceRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("AbsenceRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("AbsenceRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void AbsenceRule::onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            // 使用岗位检测器更新岗位区域（每帧一次）
            all_station_regions_ = station_detector_.getAllStations(packet->detections);
        }

        void AbsenceRule::reset()
        {
            LOG_INFO_FMT("AbsenceRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json AbsenceRule::getStatistics() const
        {
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus AbsenceRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            LOG_INFO_FMT("AbsenceRule::rule_logic()");
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            int person_not_in_station_count = 0;
            std::vector<core::InferenceResultPacket::BBox> person_not_in_station_box;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
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
                bool station_occupied = false;
                // 检查当前区域是否有人
                for (const auto &person_box : person_boxes)
                {
                    if (ZoneValidator::pointInPolygon(PixelPoint(person_box.x + person_box.w / 2, person_box.y + person_box.h / 2), station_region.region))
                    {
                        station_occupied = true;
                        break;
                    }
                }
                // 如果岗位无人且之前有人，更新离岗计数器
                if (!station_occupied)
                {
                    if (m_absent_counter_map_.find(station_id) == m_absent_counter_map_.end())
                    {
                        m_absent_counter_map_[station_id] = 1;
                    }
                    else
                    {
                        m_absent_counter_map_[station_id]++;
                    }
                    // 判断是否达到离岗阈值
                    if (m_absent_counter_map_[station_id] >= absent_threshold_)
                    {
                        core::InferenceResultPacket::BBox absent_box;
                        absent_box.x = station_region.region[0].x;
                        absent_box.y = station_region.region[0].y;
                        absent_box.w = station_region.region[1].x - station_region.region[0].x;
                        absent_box.h = station_region.region[2].y - station_region.region[0].y;
                        person_not_in_station_box.push_back(absent_box);
                        person_not_in_station_count++;
                    }
                }
                else
                {
                    // 人员回到岗位，重置离岗计数（否则计数器闩锁，防抖只生效一次）
                    m_absent_counter_map_[station_id] = 0;
                }
            }
            if (person_not_in_station_count <= 0)
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            updateZoneEvent(zone_no, packet, {});

            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("absence", AbsenceRule)
    }
}