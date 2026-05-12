// src/rules/alert/person_instrusion_rules.cpp
#include "person_intrusion_rules.h"
#include "alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream { 
    namespace rules { 
        
        PersonIntrusionRule::PersonIntrusionRule()
        {
            LOG_INFO("PersonIntrusionRule::PersonIntrusionRule()");
        }

        bool PersonIntrusionRule::initialize(const nlohmann::json& config)
        {
            LOG_INFO_FMT("PersonIntrusionRule::initialize()");
            return true;
        }

        std::vector<AlertEvent> PersonIntrusionRule::process(
            std::shared_ptr<core::InferenceResultPacket> packet,
            int64_t current_time_ms)
        {
            LOG_INFO_FMT("PersonIntrusionRule::process()");
            return std::vector<AlertEvent>();
        }

        void PersonIntrusionRule::reset()
        {
            LOG_INFO_FMT("PersonIntrusionRule::reset()");
        }

        nlohmann::json PersonIntrusionRule::getStatistics() const
        {
            LOG_INFO_FMT("PersonIntrusionRule::getStatistics()");
            return nlohmann::json();
        }

        REGISTER_ALERT_RULE("person_intrusion", PersonIntrusionRule)
    }
}
