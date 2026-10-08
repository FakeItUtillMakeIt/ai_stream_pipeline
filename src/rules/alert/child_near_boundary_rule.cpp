// src/rules/alert/child_near_boundary_rule.cpp
#include "child_near_boundary_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>

namespace ai_stream
{
    namespace rules
    {

        ChildNearBoundaryRule::ChildNearBoundaryRule()
        {
            LOG_INFO("ChildNearBoundaryRule::ChildNearBoundaryRule()");
        }

        ChildNearBoundaryRule::~ChildNearBoundaryRule() = default;

        bool ChildNearBoundaryRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("ChildNearBoundaryRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                    setName(config.value("name", ""));

                if (config.contains("child_class"))
                    child_class_ = config.value("child_class", child_class_);

                if (config.contains("boundary_classes") &&
                    config["boundary_classes"].is_array())
                {
                    boundary_classes_ = config["boundary_classes"].get<std::vector<std::string>>();
                }

                relation_threshold_ = config.value("relation_threshold", relation_threshold_);
                allow_geometric_fallback_ =
                    config.value("geometric_fallback", allow_geometric_fallback_);
                max_pixel_gap_ = config.value("max_pixel_gap", max_pixel_gap_);

                // 阈值必须落在 (0,1]，否则关系分永远不过或永远过
                if (relation_threshold_ <= 0.0f || relation_threshold_ > 1.0f)
                {
                    LOG_WARN_FMT("ChildNearBoundaryRule: relation_threshold={} out of (0,1], clamped",
                                 relation_threshold_);
                    relation_threshold_ = std::clamp(relation_threshold_, 0.01f, 1.0f);
                }
                if (boundary_classes_.empty())
                {
                    LOG_WARN("ChildNearBoundaryRule: boundary_classes is empty; the rule can never fire");
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("ChildNearBoundaryRule::initialize() exception: {}", e.what());
                return false;
            }
            return parseZones(config);
        }

        void ChildNearBoundaryRule::reset()
        {
            LOG_INFO_FMT("ChildNearBoundaryRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
            frame_hits_.clear();
        }

        nlohmann::json ChildNearBoundaryRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            return nlohmann::json{
                {"child_class", child_class_},
                {"boundary_classes", boundary_classes_},
                {"relation_threshold", relation_threshold_},
                {"geomeric_fallback", allow_geometric_fallback_},
                {"max_pixel_gap", max_pixel_gap_},
            };
        }

        void ChildNearBoundaryRule::onPreProcess(
            const std::shared_ptr<core::InferenceResultPacket> &packet)
        {
            frame_hits_.clear();
            if (!packet || packet->relations.empty())
                return;

            // 关系是有向的：只认「儿童 -> 边界」。模型给出反向的「墙 beside 儿童」
            // 不算儿童靠近墙，只判谓词会把两者混为一谈。
            for (const auto &rel : packet->relations)
            {
                if (rel.subject_class != child_class_)
                    continue;
                if (std::find(boundary_classes_.begin(), boundary_classes_.end(),
                              rel.object_class) == boundary_classes_.end())
                    continue;
                if (rel.confidence < relation_threshold_)
                    continue;
                // track_id 可能是 -1（tracker 没匹配上）。这类命中必须保留，
                // 否则整条规则在检测抖动时会永远不触发。-1 只用于"这一帧有命中"，
                // 不参与告警里的目标标识。
                frame_hits_.push_back(Hit{rel.subject_track_id, rel.predicate, rel.confidence});
            }

            // 同一帧多个谓词命中同一儿童时，只留分数最高的，避免一个人被算多次
            std::stable_sort(frame_hits_.begin(), frame_hits_.end(),
                             [](const Hit &a, const Hit &b) { return a.score > b.score; });
            // 只对已跟踪的目标去重。track_id == -1 表示"某个儿童（本帧未跟踪）"，
            // 它们之间无法用 id 区分，若也按 id 去重会把多个儿童合并成一次命中。
            std::vector<Hit> dedup;
            dedup.reserve(frame_hits_.size());
            for (const auto& h : frame_hits_) {
                if (h.track_id >= 0) {
                    auto same = std::find_if(dedup.begin(), dedup.end(),
                                             [&h](const Hit& x) {
                                                 return x.track_id == h.track_id;
                                             });
                    if (same != dedup.end()) continue;   // 已保留分数更高的
                }
                dedup.push_back(h);
            }
            frame_hits_ = std::move(dedup);
        }

        RuleStatus ChildNearBoundaryRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            (void)packet;
            if (frame_hits_.empty())
                return RuleStatus::RULE_STATUS_OK;

            // 命中项还要落在本 zone 内；zone 为空表示全域，视为都在内
            std::vector<int> track_ids;
            float best = 0.0f;
            std::string best_predicate;
            for (const auto &hit : frame_hits_)
            {
                if (!zone_points.empty())
                {
                    bool inside = false;
                    for (const auto &d : packet->detections)
                    {
                        if (d.track_id != hit.track_id)
                            continue;
                        const PixelPoint center(d.x + d.w / 2, d.y + d.h / 2);
                        inside = ZoneValidator::pointInPolygon(center, zone_points);
                        break;
                    }
                    if (!inside)
                        continue;
                }
                if (hit.track_id >= 0)
                    track_ids.push_back(hit.track_id);
                if (hit.score > best)
                {
                    best = hit.score;
                    best_predicate = hit.predicate;
                }
            }
            // 未跟踪的命中也要放行：track_id 只影响告警里能否点名具体目标，
            // 不该决定"要不要报"。要求 track_ids 非空会让规则在检测抖动时完全静默。
            if (track_ids.empty())
                LOG_DEBUG("[ChildNearBoundaryRule] hit without track_id; alerting without target id");

            updateZoneEvent(zone_no, packet, track_ids);
            LOG_INFO_FMT("[ChildNearBoundaryRule] {} near {} score={:.3f} predicate={} tracks={}",
                         child_class_, boundary_classes_.front(), best, best_predicate,
                         track_ids.size());
            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("child_near_boundary", ChildNearBoundaryRule)
    }
}
