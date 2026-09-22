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

            if (!parseZones(config))
            {
                return false;
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

            if (!packet->source_frame || !packet->source_frame->mat)
            {
                return RuleStatus::RULE_STATUS_OK;
            }
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
                        updateZoneEvent(zone_no, packet, {});
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
