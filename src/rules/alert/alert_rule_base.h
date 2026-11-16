// src/rules/alert/alert_rule_base.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <unordered_map>
#include <vector>
#include <algorithm>

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

            // ============================================================
            // 区域过滤公共工具（P0：所有规则统一 zone 语义）
            // zone 为空表示全域，不做过滤
            // ============================================================
            // 检测框中心是否落在 zone 内（大多数规则统一用中心点归属）
            static bool boxInZone(const core::InferenceResultPacket::BBox &box,
                                  const ZonePoints &zone)
            {
                if (zone.empty())
                {
                    return true;
                }
                return ZoneValidator::pointInPolygon(
                    PixelPoint(box.x + box.w / 2.0f, box.y + box.h / 2.0f), zone);
            }

            // child 框中心是否落在 parent 框内（辅助目标归属到人的常用判据）
            static bool centerInsideBox(const core::InferenceResultPacket::BBox &child,
                                        const core::InferenceResultPacket::BBox &parent)
            {
                const float cx = child.x + child.w / 2.0f;
                const float cy = child.y + child.h / 2.0f;
                return cx >= parent.x && cx <= parent.x + parent.w &&
                       cy >= parent.y && cy <= parent.y + parent.h;
            }

            // child 框面积中被 parent 覆盖的比例（0..1），用于“辅助目标归属到人”
            static float overlapRatioInParent(const core::InferenceResultPacket::BBox &child,
                                              const core::InferenceResultPacket::BBox &parent)
            {
                const float x1 = std::max(static_cast<float>(child.x), static_cast<float>(parent.x));
                const float y1 = std::max(static_cast<float>(child.y), static_cast<float>(parent.y));
                const float x2 = std::min(static_cast<float>(child.x + child.w), static_cast<float>(parent.x + parent.w));
                const float y2 = std::min(static_cast<float>(child.y + child.h), static_cast<float>(parent.y + parent.h));
                if (x2 <= x1 || y2 <= y1)
                {
                    return 0.0f;
                }
                const float child_area = static_cast<float>(child.w) * static_cast<float>(child.h);
                if (child_area <= 0.0f)
                {
                    return 0.0f;
                }
                return ((x2 - x1) * (y2 - y1)) / child_area;
            }

            // 两个轴对齐框的 IoU（0..1）
            static float boxIoU(const core::InferenceResultPacket::BBox &a,
                                const core::InferenceResultPacket::BBox &b)
            {
                const float x1 = std::max(static_cast<float>(a.x), static_cast<float>(b.x));
                const float y1 = std::max(static_cast<float>(a.y), static_cast<float>(b.y));
                const float x2 = std::min(static_cast<float>(a.x + a.w), static_cast<float>(b.x + b.w));
                const float y2 = std::min(static_cast<float>(a.y + a.h), static_cast<float>(b.y + b.h));
                if (x2 <= x1 || y2 <= y1)
                {
                    return 0.0f;
                }
                const float inter = (x2 - x1) * (y2 - y1);
                const float area_a = static_cast<float>(a.w) * static_cast<float>(a.h);
                const float area_b = static_cast<float>(b.w) * static_cast<float>(b.h);
                const float uni = area_a + area_b - inter;
                return uni > 0.0f ? inter / uni : 0.0f;
            }

            // 过滤 track_id 集合：仅保留“本帧有该 track 检测且中心在 zone 内”的目标
            static std::vector<int> filterIdsInZone(
                const std::shared_ptr<core::InferenceResultPacket> &packet,
                const ZonePoints &zone, const std::vector<int> &ids)
            {
                if (zone.empty())
                {
                    return ids;
                }
                std::vector<int> out;
                out.reserve(ids.size());
                for (int id : ids)
                {
                    for (const auto &det : packet->detections)
                    {
                        if (det.track_id == id)
                        {
                            if (boxInZone(det, zone))
                            {
                                out.push_back(id);
                            }
                            break;
                        }
                    }
                }
                return out;
            }

            // zone 内当前是否有 person（用于无法定位具体 track 的动作类规则）
            static bool anyPersonInZone(
                const std::shared_ptr<core::InferenceResultPacket> &packet,
                const ZonePoints &zone)
            {
                for (const auto &det : packet->detections)
                {
                    if (det.class_name == "person" && boxInZone(det, zone))
                    {
                        return true;
                    }
                }
                return false;
            }

            // 多边形（如岗位区域）中心点
            static PixelPoint zoneCentroid(const ZonePoints &poly)
            {
                PixelPoint c(0.0f, 0.0f);
                if (poly.empty())
                {
                    return c;
                }
                for (const auto &p : poly)
                {
                    c.x += p.x;
                    c.y += p.y;
                }
                c.x /= static_cast<float>(poly.size());
                c.y /= static_cast<float>(poly.size());
                return c;
            }

            // 多边形整体是否落在 zone 内（用中心点代表，用于岗位类区域过滤）
            static bool polygonInZone(const ZonePoints &target, const ZonePoints &zone)
            {
                if (zone.empty())
                {
                    return true;
                }
                return ZoneValidator::pointInPolygon(zoneCentroid(target), zone);
            }

            static bool hasPositiveId(const std::vector<int> &ids)
            {
                for (int id : ids)
                {
                    if (id >= 0)
                    {
                        return true;
                    }
                }
                return false;
            }

            static bool hasCommonPositiveId(const std::vector<int> &a,
                                            const std::vector<int> &b)
            {
                for (int x : a)
                {
                    if (x < 0)
                    {
                        continue;
                    }
                    if (std::find(b.begin(), b.end(), x) != b.end())
                    {
                        return true;
                    }
                }
                return false;
            }

            // 解析并校验 config["rule_zones"]，填充 intrusion_zones_/valid_intrusion_zones_
            bool parseZones(const nlohmann::json &config)
            {
                // 通用确认参数：所有规则统一在此解析，避免各规则重复/遗漏
                // （规则若已自行解析，此处结果相同）
                if (config.contains("alert_duration_ms") &&
                    config["alert_duration_ms"].is_number_integer())
                {
                    int64_t v = config["alert_duration_ms"].get<int64_t>();
                    alert_duration_ms_ = v > 0 ? static_cast<uint64_t>(v) : 0;
                }
                if (config.contains("max_disappear_count") &&
                    config["max_disappear_count"].is_number_integer())
                {
                    int v = config["max_disappear_count"].get<int>();
                    if (v > 0 && v <= 255)
                    {
                        max_disappear_count_ = static_cast<uint8_t>(v);
                    }
                    else
                    {
                        LOG_WARN_FMT("AlertRuleBase: max_disappear_count={} out of (0,255], kept {}",
                                     v, static_cast<int>(max_disappear_count_));
                    }
                }

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
                    // S3：目标集合完全切换时重置累积，避免不同目标“接力”攒时长导致误报。
                    // 仅当新旧集合都含有效 track_id 且无交集时判断为切换；
                    // 全 -1（未跟踪）时无法区分目标，保持原行为。
                    if (hasPositiveId(object_ids) &&
                        hasPositiveId(alert_target.object_ids) &&
                        !hasCommonPositiveId(object_ids, alert_target.object_ids))
                    {
                        alert_target.detect_ms = timestamp_ms;
                        alert_target.duration_ms = 0;
                        alert_target.status = AlertStatus::ALERT_STATUS_DEFAULT;
                    }
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
