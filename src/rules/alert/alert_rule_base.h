// src/rules/alert/alert_rule_base.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <unordered_map>
#include <vector>

namespace ai_stream
{
    namespace rules
    {
        /**
         * @brief 告警规则公共基类
         *
         * 收敛各规则重复的：
         *  - zone 解析/校验（parseZones）
         *  - process() 模板：加锁 → 每帧钩子 → 逐 zone 调用 rule_logic → 事件聚合/衰减
         *  - 区域事件创建/更新（updateZoneEvent）
         *
         * 派生类只需实现 rule_logic()（特征判定）、initialize()（含 parseZones）、
         * getType()/getAlertItemType()，并按需重写 onPreProcess()。
         */
        class AlertRuleBase : public IAlertRule
        {
        public:
            RuleStatus process(
                std::shared_ptr<core::InferenceResultPacket> packet,
                AlertResult &alert_result,
                int64_t current_time_ms) override
            {
                (void)current_time_ms;
                std::lock_guard<std::mutex> lock(mutex_);
                if (!packet)
                    return RuleStatus::RULE_STATUS_FAIL;

                skip_finalize_ = false;
                onPreProcess(packet);

                if (valid_intrusion_zones_.empty())
                {
                    rule_logic(packet, global_zone_no_, {});
                }
                else
                {
                    for (const auto &zone : valid_intrusion_zones_)
                    {
                        rule_logic(packet, zone.first, zone.second);
                    }
                }

                // 无有效输入帧（如无岗位/无源帧）不推进告警状态机，保持原语义
                if (!skip_finalize_)
                {
                    finalizeEvents(alert_result);
                }
                return RuleStatus::RULE_STATUS_OK;
            }

        protected:
            // 本帧是否跳过事件聚合/衰减（派生类在 onPreProcess/rule_logic 中置位）
            bool skip_finalize_ = false;

            // 每帧、进入 zone 循环前的钩子（如运行有状态检测器一次）；默认空实现
            virtual void onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
            {
                (void)packet;
            }

            // 解析并校验 config["rule_zones"]，填充 intrusion_zones_/valid_intrusion_zones_
            bool parseZones(const nlohmann::json &config)
            {
                try
                {
                    if (config.contains("rule_zones") && config["rule_zones"].is_array())
                    {
                        for (size_t i = 0; i < config["rule_zones"].size(); i++)
                        {
                            for (size_t k = 0; k < config["rule_zones"][i].size(); k++)
                            {
                                LOG_INFO_FMT("Rule zone {} add point {}: [{}, {}]", int(i + 1), int(k + 1),
                                             config["rule_zones"][i][k][0].get<float>(),
                                             config["rule_zones"][i][k][1].get<float>());
                                intrusion_zones_[uint8_t(i + 1)].push_back(PixelPoint(
                                    config["rule_zones"][i][k][0].get<float>(),
                                    config["rule_zones"][i][k][1].get<float>()));
                            }
                        }
                    }
                }
                catch (const std::exception &e)
                {
                    LOG_WARN_FMT("AlertRuleBase::parseZones exception: {}", e.what());
                    return false;
                }

                for (const auto &det_zone : intrusion_zones_)
                {
                    if (!ZoneValidator::zoneIsValid(det_zone.second))
                    {
                        LOG_INFO_FMT("AlertRuleBase: zone {} is invalid", det_zone.first);
                        continue;
                    }
                    valid_intrusion_zones_[det_zone.first] = det_zone.second;
                }
                // 如果所有区域都无效，则全域监测(不进行区域过滤)
                if (valid_intrusion_zones_.empty())
                {
                    LOG_INFO("AlertRuleBase: all zones are invalid, global monitoring");
                }
                return true;
            }

            // 区域级事件创建/更新（rule_logic 内使用，避免重复样板）
            void updateZoneEvent(uint8_t zone_no,
                                 const std::shared_ptr<core::InferenceResultPacket> &packet,
                                 const std::vector<int> &object_ids)
            {
                updateZoneEventAt(zone_no, packet->timestamp_ms, object_ids);
            }

            void updateZoneEventAt(uint8_t zone_no, int64_t timestamp_ms,
                                   const std::vector<int> &object_ids)
            {
                auto it = zone_alert_map_.find(zone_no);
                if (it == zone_alert_map_.end())
                {
                    auto alert_target = AlertEvent();
                    alert_target.detect_ms = timestamp_ms;
                    alert_target.zone_no = zone_no;
                    alert_target.non_update_count = 0;
                    alert_target.duration_ms = 0;
                    alert_target.object_ids = object_ids;
                    zone_alert_map_.insert(std::make_pair(zone_no, alert_target));
                }
                else
                {
                    auto &alert_target = it->second;
                    alert_target.non_update_count = 0;
                    alert_target.duration_ms = timestamp_ms - alert_target.detect_ms;
                    alert_target.object_ids = object_ids;
                }
            }

        private:
            // 事件聚合 + 状态机衰减（原各规则 process() 尾部的重复循环）
            void finalizeEvents(AlertResult &alert_result)
            {
                for (auto it = zone_alert_map_.begin(); it != zone_alert_map_.end(); it++)
                {
                    if (it->second.status != AlertStatus::ALERT_STATUS_OCCUR &&
                        it->second.status != AlertStatus::ALERT_STATUS_LAST &&
                        it->second.status != AlertStatus::ALERT_STATUS_END)
                    {
                        continue;
                    }
                    alert_result.alert_events.push_back(it->second);
                    alert_result.alert_count++;
                }

                for (auto it = zone_alert_map_.begin(); it != zone_alert_map_.end();)
                {
                    it->second.non_update_count++;
                    if (it->second.non_update_count > max_disappear_count_)
                    {
                        it = zone_alert_map_.erase(it);
                        continue;
                    }
                    if (it->second.status == AlertStatus::ALERT_STATUS_OCCUR)
                    {
                        it->second.status = AlertStatus::ALERT_STATUS_LAST;
                    }
                    if (it->second.status == AlertStatus::ALERT_STATUS_END)
                    {
                        it->second.status = AlertStatus::ALERT_STATUS_DEFAULT;
                    }
                    if (it->second.duration_ms > alert_duration_ms_ &&
                        it->second.status == AlertStatus::ALERT_STATUS_DEFAULT)
                    {
                        it->second.status = AlertStatus::ALERT_STATUS_OCCUR;
                        it->second.alert_name = getName();
                        it->second.alert_type = getType();
                        it->second.alert_item_type = getAlertItemType();
                    }
                    if (it->second.status != AlertStatus::ALERT_STATUS_DEFAULT &&
                        it->second.non_update_count == max_disappear_count_)
                    {
                        it->second.status = AlertStatus::ALERT_STATUS_END;
                    }
                    it->second.description = getName() + alert_status_map[it->second.status];
                    it++;
                }
            }
        };
    }
}
