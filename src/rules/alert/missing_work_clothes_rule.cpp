// src/rules/alert/missing_work_clothes_rule.cpp
#include "missing_work_clothes_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>
#include <cmath>

namespace ai_stream
{
    namespace rules
    {
        namespace
        {
            using BBox = core::InferenceResultPacket::BBox;

            // 轴对齐矩形：交集面积占人体框的比例（以 person 为分母，对衣服框略大/略偏更鲁棒）
            float interRatioOfPerson(const BBox &other, const BBox &person)
            {
                const int x1 = std::max(other.x, person.x);
                const int y1 = std::max(other.y, person.y);
                const int x2 = std::min(other.x + other.w, person.x + person.w);
                const int y2 = std::min(other.y + other.h, person.y + person.h);
                if (x2 <= x1 || y2 <= y1)
                    return 0.0f;
                const float person_area = static_cast<float>(person.w) * static_cast<float>(person.h);
                if (person_area <= 0.0f)
                    return 0.0f;
                const float inter = static_cast<float>(x2 - x1) * static_cast<float>(y2 - y1);
                return inter / person_area;
            }

            // other 的中心是否落在 person 框内（用于把 head/helmet/clothes 绑定到同一个人）
            bool centerInside(const BBox &other, const BBox &person)
            {
                const float cx = other.x + other.w / 2.0f;
                const float cy = other.y + other.h / 2.0f;
                return cx >= person.x && cx <= person.x + person.w &&
                       cy >= person.y && cy <= person.y + person.h;
            }
        } // namespace

        MissingWorkClothesRule::MissingWorkClothesRule()
        {
            LOG_INFO("MissingWorkClothesRule::MissingWorkClothesRule()");
        }

        bool MissingWorkClothesRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("MissingWorkClothesRule::initialize()");
            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }
                if (config.contains("cover_ratio") && config["cover_ratio"].is_number())
                {
                    cover_ratio_ = config["cover_ratio"].get<float>();
                }
                if (config.contains("min_person_h") && config["min_person_h"].is_number_integer())
                {
                    min_person_h_ = config["min_person_h"].get<int>();
                }
                if (config.contains("min_conf") && config["min_conf"].is_number())
                {
                    min_conf_ = config["min_conf"].get<float>();
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("MissingWorkClothesRule::initialize() exception: {}", e.what());
                return false;
            }
            LOG_INFO_FMT("MissingWorkClothesRule::initialize() cover_ratio={}, min_person_h={}, min_conf={}",
                         cover_ratio_, min_person_h_, min_conf_);
            return parseZones(config);
        }

        void MissingWorkClothesRule::reset()
        {
            LOG_INFO_FMT("MissingWorkClothesRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
        }

        nlohmann::json MissingWorkClothesRule::getStatistics() const
        {
            std::lock_guard<std::mutex> lock(mutex_);
            nlohmann::json stats;
            stats["active_alerts"] = zone_alert_map_.size();
            return stats;
        }

        RuleStatus MissingWorkClothesRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {
            (void)zone_points;
            std::vector<BBox> person_boxes;
            std::vector<BBox> head_boxes;
            std::vector<BBox> helmet_boxes;
            std::vector<BBox> work_clothes_boxes;
            for (const auto &detection : packet->detections)
            {
                if (detection.confidence < min_conf_)
                    continue;
                if (detection.class_name == "person")
                {
                    person_boxes.push_back(detection);
                }
                else if (detection.class_name == "head")
                {
                    head_boxes.push_back(detection);
                }
                else if (detection.class_name == "helmet")
                {
                    helmet_boxes.push_back(detection);
                }
                else if (detection.class_name.find("clothes") != std::string::npos ||
                         detection.class_name.find("uniform") != std::string::npos)
                {
                    work_clothes_boxes.push_back(detection);
                }
            }
            if (person_boxes.empty())
            {
                return RuleStatus::RULE_STATUS_OK;
            }

            std::vector<int> not_work_clothes_track_ids;
            for (const auto &person_box : person_boxes)
            {
                // 1) 远处/抖动的小人不做判定
                if (person_box.h < min_person_h_)
                {
                    continue;
                }
                // 2) 确认"这是一个人"：同一 person 框内必须出现人头或头盔
                bool confirmed_person = false;
                for (const auto &head : head_boxes)
                {
                    if (centerInside(head, person_box))
                    {
                        confirmed_person = true;
                        break;
                    }
                }
                if (!confirmed_person)
                {
                    for (const auto &helmet : helmet_boxes)
                    {
                        if (centerInside(helmet, person_box))
                        {
                            confirmed_person = true;
                            break;
                        }
                    }
                }
                if (!confirmed_person)
                {
                    continue;
                }
                // 3) 工作服覆盖比：取与该人绑定的工作服框中的最大覆盖
                float best_cover = 0.0f;
                for (const auto &clothes : work_clothes_boxes)
                {
                    if (!centerInside(clothes, person_box))
                        continue;
                    best_cover = std::max(best_cover, interRatioOfPerson(clothes, person_box));
                    if (best_cover >= cover_ratio_)
                        break;
                }
                // 4) 覆盖不足 => 未穿工作服
                if (best_cover < cover_ratio_)
                {
                    not_work_clothes_track_ids.push_back(person_box.track_id);
                    LOG_DEBUG_FMT("MissingWorkClothesRule: person track {} cover={:.2f} < {:.2f}, flagged",
                                  person_box.track_id, best_cover, cover_ratio_);
                }
            }

            if (!not_work_clothes_track_ids.empty())
            {
                updateZoneEvent(zone_no, packet, not_work_clothes_track_ids);
            }
            return RuleStatus::RULE_STATUS_OK;
        }
        REGISTER_ALERT_RULE("missing_work_clothes", MissingWorkClothesRule)
    }
}