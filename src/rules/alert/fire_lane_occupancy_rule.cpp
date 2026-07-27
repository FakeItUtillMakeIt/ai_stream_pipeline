// src/rules/alert/fire_lane_occupancy_rule.cpp
#include "fire_lane_occupancy_rule.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/imgcodecs.hpp>

namespace ai_stream
{
    namespace rules
    {

        FireLaneOccupancyRule::FireLaneOccupancyRule()
        {
            LOG_INFO("FireLaneOccupancyRule::FireLaneOccupancyRule()");
        }

        bool FireLaneOccupancyRule::initialize(const nlohmann::json &config)
        {
            LOG_INFO_FMT("FireLaneOccupancyRule::initialize() config: {}", config.dump().c_str());

            try
            {
                if (config.contains("name") && config["name"].is_string())
                {
                    setName(config.value("name", ""));
                }

                if (config.contains("similarity_threshold") && config["similarity_threshold"].is_number())
                {
                    similarity_threshold_ = config["similarity_threshold"].get<float>();
                }

                if (config.contains("confirm_frames") && config["confirm_frames"].is_number_integer())
                {
                    confirm_frames_ = config["confirm_frames"].get<int>();
                }

                if (config.contains("init_frames") && config["init_frames"].is_number_integer())
                {
                    init_frames_ = config["init_frames"].get<int>();
                }

                if (config.contains("update_frames") && config["update_frames"].is_number_integer())
                {
                    update_frames_ = config["update_frames"].get<int>();
                }

                if (config.contains("rule_zones") && config["rule_zones"].is_array())
                {
                    for (size_t i = 0; i < config["rule_zones"].size(); i++)
                    {
                        for (size_t k = 0; k < config["rule_zones"][i].size(); k++)
                        {
                            float x = config["rule_zones"][i][k][0].get<float>();
                            float y = config["rule_zones"][i][k][1].get<float>();
                            intrusion_zones_[uint8_t(i + 1)].push_back(PixelPoint(x, y));
                        }
                    }
                }

                if (config.contains("reference_image") && config["reference_image"].is_string())
                {
                    std::string img_path = config["reference_image"].get<std::string>();
                    if (loadReferenceImage(img_path))
                    {
                        LOG_INFO_FMT("FireLaneOccupancyRule: loaded reference image: {}", img_path);
                    }
                    else
                    {
                        LOG_WARN_FMT("FireLaneOccupancyRule: failed to load reference image: {}", img_path);
                    }
                }
            }
            catch (const std::exception &e)
            {
                LOG_WARN_FMT("FireLaneOccupancyRule::initialize() exception: {}", e.what());
                return false;
            }

            for (const auto &det_zone : intrusion_zones_)
            {
                bool zone_is_valid = ZoneValidator::zoneIsValid(det_zone.second);
                if (!zone_is_valid)
                {
                    LOG_INFO_FMT("FireLaneOccupancyRule::initialize() zone {} is invalid", det_zone.first);
                    continue;
                }
                valid_intrusion_zones_[det_zone.first] = det_zone.second;
            }

            if (valid_intrusion_zones_.empty())
            {
                LOG_INFO("FireLaneOccupancyRule::initialize() all zones are invalid, global monitoring");
            }

            LOG_INFO_FMT("FireLaneOccupancyRule::initialize() success (threshold={}, confirm={}, init={}, update={}, has_ref_img={})",
                         similarity_threshold_, confirm_frames_, init_frames_, update_frames_, has_reference_image_);
            return true;
        }

        bool FireLaneOccupancyRule::loadReferenceImage(const std::string &image_path)
        {
            reference_image_ = cv::imread(image_path);
            if (reference_image_.empty())
            {
                has_reference_image_ = false;
                return false;
            }
            has_reference_image_ = true;
            return true;
        }

        RuleStatus FireLaneOccupancyRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            AlertResult &alert_result,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("FireLaneOccupancyRule::process()");
            std::lock_guard<std::mutex> lock(mutex_);

            if (!packet || !packet->source_frame || !packet->source_frame->mat)
            {
                return RuleStatus::RULE_STATUS_FAIL;
            }

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

            uint8_t alert_count = 0;
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
                alert_count++;
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
                if (it->second.duration_ms > alert_duration_ms_ && it->second.status == AlertStatus::ALERT_STATUS_DEFAULT)
                {
                    it->second.status = AlertStatus::ALERT_STATUS_OCCUR;
                    it->second.alert_name = getName();
                    it->second.alert_type = getType();
                }
                if (it->second.status != AlertStatus::ALERT_STATUS_DEFAULT &&
                    it->second.non_update_count == max_disappear_count_)
                {
                    it->second.status = AlertStatus::ALERT_STATUS_END;
                }
                it->second.description = getName() + alert_status_map[it->second.status];
                it++;
            }

            return RuleStatus::RULE_STATUS_OK;
        }

        void FireLaneOccupancyRule::reset()
        {
            LOG_INFO("FireLaneOccupancyRule::reset()");
            std::lock_guard<std::mutex> lock(mutex_);
            zone_alert_map_.clear();
            reference_features_.clear();
            occupy_counts_.clear();
            idle_counts_.clear();
            init_counts_.clear();
            initialized_.clear();
        }

        nlohmann::json FireLaneOccupancyRule::getStatistics() const
        {
            return nlohmann::json();
        }

        RuleStatus FireLaneOccupancyRule::rule_logic(
            const std::shared_ptr<core::InferenceResultPacket> packet,
            uint8_t zone_no, ZonePoints zone_points)
        {

            const cv::Mat &frame = *packet->source_frame->mat;

            FeatureVector cur_feature = FeatureExtractor::extract(frame, zone_points);
            LOG_INFO_FMT("cur_feature.valid={}", cur_feature.valid);
            if (!cur_feature.valid)
            {
                return RuleStatus::RULE_STATUS_OK;
            }

            if (has_reference_image_ && initialized_[zone_no])
            {
                auto it = reference_features_.find(zone_no);
                if (it != reference_features_.end())
                {
                    float similarity = FeatureExtractor::computeSimilarity(it->second, cur_feature);
                    LOG_INFO_FMT("FireLaneOccupancyRule::rule_logic() zone={} similarity={}", zone_no, similarity);
                    if (similarity < similarity_threshold_)
                    {
                        occupy_counts_[zone_no]++;
                        idle_counts_[zone_no] = 0;
                    }
                    else
                    {
                        idle_counts_[zone_no]++;
                        occupy_counts_[zone_no] = 0;
                        if (idle_counts_[zone_no] > update_frames_)
                        {
                            FeatureExtractor::updateReference(it->second, cur_feature);
                            idle_counts_[zone_no] = 0;
                        }
                    }

                    if (occupy_counts_[zone_no] >= confirm_frames_)
                    {
                        auto alert_it = zone_alert_map_.find(zone_no);
                        if (alert_it == zone_alert_map_.end())
                        {
                            AlertEvent event;
                            event.detect_ms = packet->timestamp_ms;
                            event.zone_no = zone_no;
                            event.non_update_count = 0;
                            event.duration_ms = 0;
                            zone_alert_map_[zone_no] = event;
                        }
                        else
                        {
                            alert_it->second.non_update_count = 0;
                            alert_it->second.duration_ms = packet->timestamp_ms - alert_it->second.detect_ms;
                        }
                    }
                }
                return RuleStatus::RULE_STATUS_OK;
            }

            if (has_reference_image_ && !initialized_[zone_no])
            {
                FeatureVector ref_feature = FeatureExtractor::extract(reference_image_, zone_points);
                if (ref_feature.valid)
                {
                    reference_features_[zone_no] = ref_feature;
                    initialized_[zone_no] = true;
                    LOG_INFO_FMT("FireLaneOccupancyRule: zone {} initialized with reference image", zone_no);
                }
                return RuleStatus::RULE_STATUS_OK;
            }

            if (!initialized_[zone_no])
            {
                init_counts_[zone_no]++;
                FeatureExtractor::accumulateReference(reference_features_[zone_no], cur_feature, init_counts_[zone_no]);

                if (init_counts_[zone_no] >= init_frames_)
                {
                    initialized_[zone_no] = true;
                    LOG_INFO_FMT("FireLaneOccupancyRule: zone {} auto-initialized after {} frames", zone_no, init_frames_);
                }
                return RuleStatus::RULE_STATUS_OK;
            }

            auto it = reference_features_.find(zone_no);
            if (it == reference_features_.end())
            {
                return RuleStatus::RULE_STATUS_OK;
            }

            float similarity = FeatureExtractor::computeSimilarity(it->second, cur_feature);

            if (similarity < similarity_threshold_)
            {
                occupy_counts_[zone_no]++;
                idle_counts_[zone_no] = 0;
            }
            else
            {
                idle_counts_[zone_no]++;
                occupy_counts_[zone_no] = 0;
                if (idle_counts_[zone_no] > update_frames_)
                {
                    FeatureExtractor::updateReference(it->second, cur_feature);
                    idle_counts_[zone_no] = 0;
                }
            }

            if (occupy_counts_[zone_no] >= confirm_frames_)
            {
                auto alert_it = zone_alert_map_.find(zone_no);
                if (alert_it == zone_alert_map_.end())
                {
                    AlertEvent event;
                    event.detect_ms = packet->timestamp_ms;
                    event.zone_no = zone_no;
                    event.non_update_count = 0;
                    event.duration_ms = 0;
                    zone_alert_map_[zone_no] = event;
                }
                else
                {
                    alert_it->second.non_update_count = 0;
                    alert_it->second.duration_ms = packet->timestamp_ms - alert_it->second.detect_ms;
                }
            }

            return RuleStatus::RULE_STATUS_OK;
        }

        REGISTER_ALERT_RULE("fire_lane_occupancy", FireLaneOccupancyRule)

    } // namespace rules
} // namespace ai_stream
