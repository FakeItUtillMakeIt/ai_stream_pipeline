// src/rules/alert/person_loitering_rule.cpp
#include "person_loitering_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <set>
#include <vector>

namespace ai_stream
{
    namespace rules
    {
        namespace
        {
            inline uint64_t zoneTrackKey(uint8_t zone_no, int track_id)
            {
                return (static_cast<uint64_t>(zone_no) << 32) |
                       static_cast<uint32_t>(track_id);
            }
        }

        PersonLoiteringRule::PersonLoiteringRule()
        {
            LOG_INFO("PersonLoiteringRule::PersonLoiteringRule()");
        }

        bool PersonLoiteringRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("PersonLoiteringRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                    setName(config.value("name", ""));

                target_class_ = config.value("target_class", target_class_);
                loiter_duration_ms_ = config.value("loiter_duration_ms", loiter_duration_ms_);
                leave_grace_ms_ = config.value("leave_grace_ms", leave_grace_ms_);

                // 判为徘徊后，还需持续这么久才从 DEFAULT 进 OCCUR（二次确认，防抖）
                if (config.contains("alert_duration_ms"))
                    alert_duration_ms_ = static_cast<uint64_t>(config["alert_duration_ms"].get<int64_t>());
                else if (config.contains("duration_ms"))
                    alert_duration_ms_ = static_cast<uint64_t>(config["duration_ms"].get<int64_t>());

                if (loiter_duration_ms_ <= 0)
                {
                    LOG_WARN_FMT("PersonLoiteringRule: loiter_duration_ms={} <=0, clamped to 1000",
                                 loiter_duration_ms_);
                    loiter_duration_ms_ = 1000;
                }
                LOG_INFO_FMT("PersonLoiteringRule: target={} loiter_duration_ms={} leave_grace_ms={} alert_duration_ms={}",
                             target_class_, loiter_duration_ms_, leave_grace_ms_,
                             (long long)alert_duration_ms_);
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("PersonLoiteringRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void PersonLoiteringRule::reset()
        {
            LOG_INFO_FMT("PersonLoiteringRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
            first_in_zone_ms_.clear();
            last_seen_ms_.clear();
        }

        nlohmann::json PersonLoiteringRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return nlohmann::json{
                {"target_class", target_class_},
                {"loiter_duration_ms", loiter_duration_ms_},
                {"leave_grace_ms", leave_grace_ms_},
                {"tracked_persons", static_cast<int>(first_in_zone_ms_.size())}};
        }

        RuleStatus PersonLoiteringRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            if (!packet)
                return RuleStatus::RULE_STATUS_OK;

            const int64_t now = packet->timestamp_ms;
            std::vector<int> loiter_ids;
            std::set<uint64_t> present_keys;

            for (const auto &det : packet->detections)
            {
                if (det.class_name != target_class_)
                    continue;
                // 滞留时长依赖稳定跟踪 ID；无 track 无法计时，跳过
                if (det.track_id < 0)
                    continue;

                const bool in_zone = zone_points.empty()
                    ? true
                    : ZoneValidator::pointInPolygon(
                          PixelPoint(det.x + det.w / 2, det.y + det.h / 2), zone_points);
                if (!in_zone)
                    continue;

                const uint64_t key = zoneTrackKey(zone_no, det.track_id);
                present_keys.insert(key);

                auto it = first_in_zone_ms_.find(key);
                if (it == first_in_zone_ms_.end())
                {
                    first_in_zone_ms_[key] = now;   // 首次进入本区域，开始计时
                    it = first_in_zone_ms_.find(key);
                }
                last_seen_ms_[key] = now;

                const int64_t dwell = now - it->second;
                if (dwell >= loiter_duration_ms_)
                    loiter_ids.push_back(det.track_id);
            }

            // 清理：本区域中已离开(或漏检超过宽限)的 track，重置其滞留计时
            std::vector<uint64_t> to_erase;
            for (auto &kv : first_in_zone_ms_)
            {
                const uint64_t key = kv.first;
                if ((key >> 32) != static_cast<uint64_t>(zone_no))
                    continue;                       // 只处理本 zone 的键
                if (present_keys.count(key))
                    continue;                       // 这帧仍在区域内
                auto ls = last_seen_ms_.find(key);
                const int64_t last = (ls != last_seen_ms_.end()) ? ls->second : kv.second;
                if (now - last > leave_grace_ms_)
                    to_erase.push_back(key);        // 离开超过宽限 → 视为已离开，清零
            }
            for (uint64_t key : to_erase)
            {
                first_in_zone_ms_.erase(key);
                last_seen_ms_.erase(key);
            }

            if (loiter_ids.empty())
                return RuleStatus::RULE_STATUS_OK;

            updateZoneEvent(zone_no, packet, loiter_ids);
            LOG_DEBUG_FMT("PersonLoiteringRule: zone {} loitering tracks={} now={}",
                          zone_no, loiter_ids.size(), now);
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("person_loitering", PersonLoiteringRule)
    }
}
