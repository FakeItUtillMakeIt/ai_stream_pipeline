// src/rules/alert/falling_rule.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include <unordered_map>
#include <mutex>
#include <string>
#include <vector>

namespace ai_stream
{
    namespace rules
    {
        /**
         * @brief 跌倒过程告警规则
         *
         * 区别于 FallDownRule（仅按当前帧是否存在 fall_down 类别静态判定）：
         * 本规则基于 track_id 追踪同一目标的类别转移，识别「站立(person) -> 倒地(down)」
         * 的过程，并要求 down 状态持续 down_confirm_ms 后才确认跌倒，避免单帧误检。
         *
         * 依赖 TrackerNode 在类别跃迁（person->down）时保持同一 track_id。
         *
         * 告警类型复用 AlertType::FALL_DOWN。
         */
        class FallingRule : public IAlertRule
        {
        public:
            FallingRule();

            bool initialize(const nlohmann::json &config) override;
            RuleStatus process(
                std::shared_ptr<core::InferenceResultPacket> packet,
                AlertResult &alert_result,
                int64_t current_time_ms) override;
            void reset() override;
            nlohmann::json getStatistics() const override;

            AlertType getType() const override { return AlertType::FALL_DOWN; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_PERSON_BEHAVIOR; }

        private:
            RuleStatus rule_logic(
                const std::shared_ptr<core::InferenceResultPacket> packet,
                uint8_t zone_no, ZonePoints zone_points) override;

            // 单条轨迹的类别转移状态
            struct TrackFallState
            {
                bool was_person = false;    // 是否曾观测到 person
                bool in_down = false;       // 当前是否处于 down
                bool confirmed = false;     // down 已持续达标
                int64_t down_start_ms = 0;  // down 状态起始时间
                int64_t last_seen_ms = 0;   // 最近一次观测时间
            };

            bool isPersonClass(const std::string &name) const;
            bool isDownClass(const std::string &name) const;

            // 更新各轨迹的类别转移状态，并刷新本帧确认跌倒的目标集合
            void updateTrackStates(
                const std::shared_ptr<core::InferenceResultPacket> &packet,
                int64_t now_ms);
            void cleanupStaleTracks(int64_t now_ms);

            std::string person_class_ = "person";
            std::vector<std::string> down_classes_{"down"};
            int64_t down_confirm_ms_ = 1000;    // down 需持续的时长
            int64_t track_timeout_ms_ = 5000;   // 轨迹状态过期时间

            std::unordered_map<int, TrackFallState> track_states_;
            std::unordered_map<int, core::InferenceResultPacket::BBox> confirmed_falls_;
        };
    }
}
