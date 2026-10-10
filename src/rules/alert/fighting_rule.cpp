// src/rules/alert/fighting_rule.cpp
#include "fighting_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <algorithm>
#include <vector>

namespace ai_stream
{
    namespace rules
    {

        FightingRule::FightingRule() : fighting_detector_(FightingDetector::Config())
        {
            LOG_INFO("FightingRule::FightingRule()");
            action_recognition_mode_ = ActionRecongnitionType::ACTION_RECOGNITION_POSE;
        }

        bool FightingRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("FightingRule::initialize()");
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
                // fighting 一次保持时长(ms)：桥接动作模型逐窗抖动，默认 0（不保持）
                if (config.contains("fight_hold_ms"))
                    fight_hold_ms_ = config["fight_hold_ms"].get<int64_t>();
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("FightingRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void FightingRule::onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            last_is_fighting_ = false;
            last_fight_track_ids_.clear();

            // MODEL 模式：只依据 VideoMAE 的动作结果判定，不依赖 person 框。
            // （旧实现把 person_boxes.empty() 提前 return 挡在 MODEL 分支之前，
            //  导致没检出 person 的帧即便动作是 fighting 也被跳过。）
            if (action_recognition_mode_ == ActionRecongnitionType::ACTION_RECOGNITION_MODEL)
            {
                bool now_fight = false;
                for (const auto &action_result : packet->action_results)
                {
                    LOG_INFO_FMT("FightingRule::onPreProcess() action_result.action_label: {}", action_result.action_label);
                    if (action_result.action_label == alertTypeMap[AlertType::FIGHTING])
                        now_fight = true;
                }
                const int64_t ts = packet->timestamp_ms;
                if (now_fight)
                    last_fighting_ts_ = ts;
                // 一次保持：桥接 VideoMAE 逐窗 fighting/other 抖动，让 duration 能连续累积
                if (fight_hold_ms_ > 0 && last_fighting_ts_ >= 0)
                    last_is_fighting_ = (ts - last_fighting_ts_) <= fight_hold_ms_;
                else
                    last_is_fighting_ = now_fight;
                return;
            }

            // POSE 模式：需要 person 框做肢体/轨迹启发式
            std::vector<ai_stream::core::InferenceResultPacket::BBox> person_boxes;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name == "person")
                    person_boxes.push_back(detection);
            }
            LOG_INFO_FMT("FightingRule::onPreProcess() person_boxes.size(): {}", person_boxes.size());
            if (person_boxes.empty())
            {
                return;
            }
            auto fight_result = fighting_detector_.process(person_boxes);
            last_is_fighting_ = fight_result.is_fighting;
            last_fight_track_ids_ = fight_result.active_track_ids;
        }

        void FightingRule::reset()
        {
            LOG_INFO_FMT("FightingRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json FightingRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            LOG_INFO_FMT("FightingRule::getStatistics()");
            return nlohmann::json();
        }

        RuleStatus FightingRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            // 统计 zone 内人数（保持“至少2人”语义）；检测结果已在本帧 process() 计算一次
            int person_count = 0;
            std::vector<int> in_zone_track_ids;
            for (const auto &detection : packet->detections)
            {
                if (detection.class_name != "person")
                    continue;
                bool in_zone = zone_points.empty() ? true : ZoneValidator::pointInPolygon(PixelPoint(detection.x + detection.w / 2, detection.y + detection.h / 2), zone_points);
                if (in_zone)
                {
                    person_count++;
                    if (detection.track_id > 0)
                        in_zone_track_ids.push_back(detection.track_id);
                }
            }
            if (person_count < 2 || !last_is_fighting_)
                return RuleStatus::RULE_STATUS_OK;

            std::vector<int> fight_ids;
            if (action_recognition_mode_ == ActionRecongnitionType::ACTION_RECOGNITION_POSE)
            {
                if (last_fight_track_ids_.empty())
                {
                    fight_ids = in_zone_track_ids;
                }
                else
                {
                    for (int id : last_fight_track_ids_)
                    {
                        if (std::find(in_zone_track_ids.begin(), in_zone_track_ids.end(), id) != in_zone_track_ids.end())
                            fight_ids.push_back(id);
                    }
                    if (fight_ids.size() < 2)
                        return RuleStatus::RULE_STATUS_OK;
                }
            }

            updateZoneEvent(zone_no, packet, fight_ids);
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("fighting", FightingRule)
    }
}
