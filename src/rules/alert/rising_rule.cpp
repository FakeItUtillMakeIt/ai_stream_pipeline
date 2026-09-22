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
                LOG_INFO_FMT("RisingRule::initialize() config: {}", config.dump().c_str());
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
                if (config.contains("rule_zones") && config["rule_zones"].is_array())
                {
                    for (size_t i = 0; i < config["rule_zones"].size(); i++)
                    {
                        for (size_t k = 0; k < config["rule_zones"][i].size(); k++)
                        {
                            LOG_INFO_FMT("Rule zone {} add point {}: [{}, {}]", int(i + 1), int(k + 1), config["rule_zones"][i][k][0].get<float>(), config["rule_zones"][i][k][1].get<float>());
                            intrusion_zones_[uint8_t(i + 1)].push_back(PixelPoint(config["rule_zones"][i][k][0].get<float>(), config["rule_zones"][i][k][1].get<float>()));
                        }
                        
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("RisingRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("RisingRule::initialize() success");

            // 判断区域是否有效/配置
            uint8_t invaild_zone_count = 0;
            int min_gathering_thresh = std::numeric_limits<int>::max();
            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("RisingRule::initialize() zone {} is invalid", det_zone.first);
                    invaild_zone_count++;
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }
            // 如果所有区域都无效，则全域监测(不进行区域过滤)
            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("RisingRule::initialize() all zones are invalid, global monitoring");
            }

            return true;
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
            {
                auto it = zone_alert_map_.find(zone_no);
                if (it == zone_alert_map_.end())
                {
                    auto alert_target = AlertEvent();
                    alert_target.detect_ms = packet->timestamp_ms;
                    alert_target.zone_no = zone_no;
                    alert_target.non_update_count = 0;
                    alert_target.duration_ms = 0;
                    alert_target.object_ids = rising_track_ids;
                    zone_alert_map_.insert(std::make_pair(zone_no, alert_target));
                }
                else
                {
                    auto &alert_target = it->second;
                    alert_target.non_update_count = 0;
                    alert_target.duration_ms = packet->timestamp_ms - alert_target.detect_ms;
                    alert_target.object_ids = rising_track_ids;
                }
            }
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("rising", RisingRule)
    }
}
