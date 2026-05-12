// src/rules/alert/person_instrusion.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include <unordered_map>
#include <mutex>

namespace ai_stream
{
    namespace rules
    {

        class PersonIntrusionRule : public IAlertRule
        {
        public:
            PersonIntrusionRule();
       
            bool initialize(const nlohmann::json &config) override;
            std::vector<AlertEvent> process(
                std::shared_ptr<core::InferenceResultPacket> packet, 
                int64_t current_time_ms) override;

            std::string getName() const override { return alertTypeMap[getType()]; }
            AlertType getType() const override { return AlertType::PERSON_INTRUSION; }
            void reset() override;
            nlohmann::json getStatistics() const override;

        private:
            struct IntrusionState
            {
                int64_t first_in_time_ms;
                int intrusion_count;
            };
            std::map<std::string,std::vector<std::pair<float, float>>> intrusion_zones_;// 区域列表
            std::unordered_map<std::string, IntrusionState> area_intrusion_states_; // 每个区域详情
            mutable std::mutex mutex_;
            
        };
    }
}