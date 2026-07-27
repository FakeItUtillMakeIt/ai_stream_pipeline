// src/rules/alert/fire_lane_occupancy_rule.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include "detector/feature_extractor.h"
#include <unordered_map>
#include <mutex>
#include <opencv2/core/mat.hpp>

namespace ai_stream
{
    namespace rules
    {

        class FireLaneOccupancyRule : public IAlertRule
        {
        public:
            FireLaneOccupancyRule();

            bool initialize(const nlohmann::json &config) override;
            RuleStatus process(
                std::shared_ptr<core::InferenceResultPacket> packet,
                AlertResult &alert_result,
                int64_t current_time_ms = 0) override;

            AlertType getType() const override { return AlertType::FIRE_LANE_OCCUPANCY; }
            void reset() override;
            nlohmann::json getStatistics() const override;

            bool loadReferenceImage(const std::string &image_path);

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet,
                                  uint8_t zone_no, ZonePoints zone_points) override;

            float similarity_threshold_ = 0.75f;
            int confirm_frames_ = 30;
            int init_frames_ = 300;
            int update_frames_ = 300;

            std::map<uint8_t, FeatureVector> reference_features_;
            std::map<uint8_t, int> occupy_counts_;
            std::map<uint8_t, int> idle_counts_;
            std::map<uint8_t, int> init_counts_;
            std::map<uint8_t, bool> initialized_;
            std::map<uint8_t, ZonePoints> zone_masks_;

            cv::Mat reference_image_;
            bool has_reference_image_ = false;
        };

    } // namespace rules
} // namespace ai_stream
