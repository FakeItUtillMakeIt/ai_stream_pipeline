// src/rules/alert/child_near_boundary_rule.h
#pragma once

#include "alert_rule_base.h"

#include <memory>
#include <string>
#include <vector>

namespace ai_stream
{
    namespace rules
    {

        /**
         * @brief 儿童靠近墙/大门告警
         *
         * 判定来自视觉关系节点写入的 packet->relations，而不是几何距离：
         * 「靠近」这件事本身由场景图模型判断，本规则只做三件收口的事——
         * 方向、类别、区域。
         *
         * **方向必须显式判**：关系是有向的，模型可能给出「墙 beside 儿童」，
         * 只看谓词会把墙靠人误报成儿童靠近墙。
         */
        class ChildNearBoundaryRule : public AlertRuleBase
        {
        public:
            ChildNearBoundaryRule();
            ~ChildNearBoundaryRule() override;

            bool initialize(const nlohmann::json &config) override;

            AlertType getType() const override { return AlertType::CHILD_NEAR_BOUNDARY; }
            AlertItemType getAlertItemType() const override { return AlertItemType::ITEM_SCENE_RECOGNITION; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        protected:
            void onPreProcess(const std::shared_ptr<core::InferenceResultPacket> &packet) override;

        private:
            RuleStatus rule_logic(const std::shared_ptr<core::InferenceResultPacket> packet,
                                  uint8_t zone_no, ZonePoints zone_points) override;

        private:
            //! 主体类别（儿童），默认为 child
            std::string child_class_ = "child";
            //! 客体类别（边界），默认 wall / gate
            std::vector<std::string> boundary_classes_{"wall", "gate"};
            //! 关系分门限
            float relation_threshold_ = 0.2f;
            //! 关系节点尚未出结果时，是否回退到几何判定。默认关闭：
            //! 回退会让"关系模型没跑起来"看起来像"一切正常"，那更危险。
            bool allow_geometric_fallback_ = false;
            //! 几何回退的像素间距
            float max_pixel_gap_ = 60.0f;

            //! 本帧命中的 (child_track_id, predicate, score, zone 内)
            struct Hit
            {
                int track_id = -1;
                std::string predicate;
                float score = 0.0f;
            };
            //! 每帧只判一次，各 zone 复用（关系节点已经跑过了，这里只是筛方向）
            std::vector<Hit> frame_hits_;
        };
    }
}
