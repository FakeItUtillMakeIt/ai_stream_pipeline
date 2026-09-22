// src/rules/alert/fire_lane_occupancy_rule.h
#pragma once

#include "alert_rule_base.h"
#include "detector/feature_extractor.h"
#include <unordered_map>
#include <mutex>
#include <opencv2/core/mat.hpp>

namespace ai_stream
{
    namespace rules
    {

        class FireLaneOccupancyRule : public AlertRuleBase
        {
        public:
            FireLaneOccupancyRule();

            bool initialize(const nlohmann::json &config) override;

            AlertType getType() const override { return AlertType::FIRE_LANE_OCCUPANCY; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_SCENE_RECOGNITION; }
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
