// src/rules/alert/falling_rule.cpp
#include "falling_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream
{
    namespace rules
    {

        FallingRule::FallingRule()
        {
            LOG_INFO("FallingRule::FallingRule()");
            // 跌倒确认由 down 状态的持续时间（down_confirm_ms_）把关，
            // 此处复用基类的告警持续窗口，设为较小值以便确认后尽快上报
            alert_duration_ms_ = 100;
        }

        bool FallingRule::isPersonClass(const std::string &name) const
        {
            return name == person_class_;
        }

        bool FallingRule::isDownClass(const std::string &name) const
        {
            for (const auto &cls : down_classes_)
            {
                if (name == cls)
                {
                    return true;
                }
            }
            return false;
        }

        bool FallingRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("FallingRule::initialize()");
            try
            {
                LOG_INFO_FMT("FallingRule::initialize() config: {}", config.dump().c_str());
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
                if (config.contains("person_class") && config["person_class"].is_string())
                {
                    person_class_ = config["person_class"].get<std::string>();
                }
                if (config.contains("down_class"))
                {
                    std::vector<std::string> down_classes;
                    if (config["down_class"].is_string())
                    {
                        down_classes.push_back(config["down_class"].get<std::string>());
                    }
                    else if (config["down_class"].is_array())
                    {
                        for (const auto &cls : config["down_class"])
                        {
                            if (cls.is_string())
                            {
                                down_classes.push_back(cls.get<std::string>());
                            }
                        }
                    }
                    if (!down_classes.empty())
                    {
                        down_classes_ = std::move(down_classes);
                    }
                }
                if (config.contains("down_confirm_ms") && config["down_confirm_ms"].is_number_integer())
                {
                    down_confirm_ms_ = config["down_confirm_ms"].get<int64_t>();
                }
                if (config.contains("track_timeout_ms") && config["track_timeout_ms"].is_number_integer())
                {
                    track_timeout_ms_ = config["track_timeout_ms"].get<int64_t>();
                }
                if (config.contains("alert_duration_ms") && config["alert_duration_ms"].is_number_integer())
                {
                    alert_duration_ms_ = static_cast<uint64_t>(config["alert_duration_ms"].get<int64_t>());
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("FallingRule::initialize() exception: {}", e.what());
                return false;
            }

            LOG_INFO_FMT("FallingRule::initialize() person_class={}, down_classes={}, down_confirm_ms={}, track_timeout_ms={}, alert_duration_ms={}",
                         person_class_, down_classes_.size(), down_confirm_ms_, track_timeout_ms_, alert_duration_ms_);

            return parseZones(config);
        }

        void FallingRule::onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            // 更新轨迹类别转移状态（每帧仅一次）
            const int64_t now_ms = packet->timestamp_ms;
            confirmed_falls_.clear();
            updateTrackStates(packet, now_ms);
            cleanupStaleTracks(now_ms);
        }

        void FallingRule::updateTrackStates(
            const std::shared_ptr<core::InferenceResultPacket> &packet,
            int64_t now_ms)
        {
            for (const auto &detection : packet->detections)
            {
                if (detection.track_id < 0)
                {
                    continue;
                }

                if (isPersonClass(detection.class_name))
                {
                    auto &state = track_states_[detection.track_id];
                    state.was_person = true;
                    state.last_seen_ms = now_ms;
                    // 由 down 恢复站立：取消跌倒判定
                    if (state.in_down)
                    {
                        LOG_INFO_FMT("[FallingRule] track {} recovered to person, cancel falling", detection.track_id);
                        state.in_down = false;
                        state.confirmed = false;
                        state.down_start_ms = 0;
                    }
                }
                else if (isDownClass(detection.class_name))
                {
                    auto it = track_states_.find(detection.track_id);
                    // 未观测过 person 的 down 不认定为跌倒过程（要求完整转移过程）
                    if (it == track_states_.end() || !it->second.was_person)
                    {
                        continue;
                    }
                    auto &state = it->second;
                    if (!state.in_down)
                    {
                        state.in_down = true;
                        state.confirmed = false;
                        state.down_start_ms = now_ms;
                        LOG_INFO_FMT("[FallingRule] track {} transition {} -> {} at {}ms",
                                     detection.track_id, person_class_, detection.class_name, now_ms);
                    }
                    state.last_seen_ms = now_ms;
                    if (!state.confirmed && (now_ms - state.down_start_ms) >= down_confirm_ms_)
                    {
                        state.confirmed = true;
                        LOG_INFO_FMT("[FallingRule] track {} confirmed falling (down {}ms)",
                                     detection.track_id, now_ms - state.down_start_ms);
                    }
                    if (state.confirmed)
                    {
                        confirmed_falls_[detection.track_id] = detection;
                    }
                }
            }
        }

        void FallingRule::cleanupStaleTracks(int64_t now_ms)
        {
            for (auto it = track_states_.begin(); it != track_states_.end();)
            {
                if (now_ms - it->second.last_seen_ms > track_timeout_ms_)
                {
                    it = track_states_.erase(it);
                }
                else
                {
                    ++it;
                }
            }
        }

        void FallingRule::reset()
        {
            LOG_INFO_FMT("FallingRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
            track_states_.clear();
            confirmed_falls_.clear();
        }

        nlohmann::json FallingRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            stats["tracking_tracks"] = track_states_.size();
            return stats;
        }

        RuleStatus FallingRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            std::vector<int> fall_track_ids;
            for (const auto &entry : confirmed_falls_)
            {
                const auto &box = entry.second;
                if (!zone_points.empty())
                {
                    PixelPoint center(box.x + box.w / 2, box.y + box.h / 2);
                    if (!ZoneValidator::pointInPolygon(center, zone_points))
                    {
                        continue;
                    }
                }
                fall_track_ids.push_back(entry.first);
            }

            if (fall_track_ids.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }

            updateZoneEvent(zone_no, packet, fall_track_ids);
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("falling", FallingRule)
    }
}
