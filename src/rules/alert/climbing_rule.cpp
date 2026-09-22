// src/rules/alert/climbing_rule.cpp
#include "climbing_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        ClimbingRule::ClimbingRule() : climbing_detector_(ClimbingDetector::Config())
        {
            LOG_INFO("ClimbingRule::ClimbingRule()");
            action_recognition_mode_ = ActionRecongnitionType::ACTION_RECOGNITION_POSE;
            alert_duration_ms_ = 100;
        }

        bool ClimbingRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("ClimbingRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
                if (config.contains("action_recongnition_mode") && config["action_recongnition_mode"].is_string())
                {
                    std::string mode = config.value("action_recongnition_mode", "");
                    if (mode == "model")
                    {
                        action_recognition_mode_ = ActionRecongnitionType::ACTION_RECOGNITION_MODEL;
                    }
                    else if (mode == "pose")
                    {
                        action_recognition_mode_ = ActionRecongnitionType::ACTION_RECOGNITION_POSE;
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("ClimbingRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void ClimbingRule::onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            // 每帧只运行一次检测器/模型判定（多 zone 时不得重复推进状态机）
            last_is_climbing_ = false;
            last_climb_track_ids_.clear();
            std::vector<ai_stream::core::InferenceResultPacket::BBox> person_boxes;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                    person_boxes.push_back(detection);
            }
            if (person_boxes.empty())
            {
                return;
            }
            if (action_recognition_mode_ == ActionRecongnitionType::ACTION_RECOGNITION_MODEL)
            {
                for (const auto &action_result : packet->action_results)
                {
                    if (action_result.action_label == alertTypeMap[AlertType::CLIMBING])
                        last_is_climbing_ = true;
                }
            }
            else if (action_recognition_mode_ == ActionRecongnitionType::ACTION_RECOGNITION_POSE)
            {
                auto climb_result = climbing_detector_.process(person_boxes);
                last_is_climbing_ = climb_result.is_climbing;
                last_climb_track_ids_ = climb_result.active_track_ids;
            }
        }

        void ClimbingRule::reset()
        {
            LOG_INFO_FMT("ClimbingRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json ClimbingRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("ClimbingRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus ClimbingRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            // 检测结果已在本帧 process() 中计算一次，这里只做 zone 归属聚合
            if (!last_is_climbing_)
            {
                return RuleStatus::RULE_STATUS_OK;
            }
            updateZoneEvent(zone_no, packet, last_climb_track_ids_);
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("climbing", ClimbingRule)
        // 兼容历史拼写错误的旧配置名
        REGISTER_ALERT_RULE("clambing", ClimbingRule)
    }
}
