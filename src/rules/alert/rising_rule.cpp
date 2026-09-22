// src/rules/alert/rising_rule.cpp
#include "rising_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        RisingRule::RisingRule() : rising_detector_()
        {
            LOG_INFO("RisingRule::RisingRule()");
            alert_duration_ms_ = 100;
        }

        bool RisingRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("RisingRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("RisingRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void RisingRule::onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            // 首次拿到源帧时按真实分辨率设置检测器归一化基准（默认 1080p 会致灵敏度漂移）
            if (!resolution_set_ && packet->source_frame &&
                packet->source_frame->width > 0 && packet->source_frame->height > 0)
            {
                rising_detector_.set_video_resolution(packet->source_frame->width,
                                                      packet->source_frame->height);
                resolution_set_ = true;
                LOG_INFO_FMT("[RisingRule] detector resolution set to {}x{}",
                             packet->source_frame->width, packet->source_frame->height);
            }

            // 每帧只运行一次检测器（多 zone 时不得重复推进状态机）
            std::vector<core::InferenceResultPacket::BBox> person_boxes;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                    person_boxes.push_back(detection);
            }
            last_rising_result_ = rising_detector_.process(person_boxes, packet->frame_id);
        }

        void RisingRule::reset()
        {
            LOG_INFO_FMT("RisingRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json RisingRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("RisingRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus RisingRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            // 检测结果已在本帧 process() 中计算一次，这里只做 zone 归属聚合
            if (!last_rising_result_.is_rising)
            {
                return RuleStatus::RULE_STATUS_OK;
            }

            std::vector<int> rising_track_ids = last_rising_result_.active_track_ids;
            updateZoneEvent(zone_no, packet, rising_track_ids);
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("rising", RisingRule)
    }
}
