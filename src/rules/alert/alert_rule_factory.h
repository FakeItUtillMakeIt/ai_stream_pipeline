// src/rules/alert/alert_rule_factory.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include <functional>
#include <map>
#include <memory>
#include <string>

namespace ai_stream {
namespace rules {

using AlertRuleCreator = std::function<AlertRulePtr()>;

class AlertRuleFactory {
public:
    static AlertRuleFactory& instance() {
        static AlertRuleFactory factory;
        return factory;
    }

    void registerCreator(const std::string& type, AlertRuleCreator creator) {
        creators_[type] = std::move(creator);
    }

    AlertRulePtr create(const std::string& type) {
        auto it = creators_.find(type);
        if (it != creators_.end()) return it->second();
        return nullptr;
    }

private:
    AlertRuleFactory() = default;
    std::map<std::string, AlertRuleCreator> creators_;
};

#define REGISTER_ALERT_RULE(type, class_name) \
    static struct AlertRuleRegistrar_##class_name { \
        AlertRuleRegistrar_##class_name() { \
            AlertRuleFactory::instance().registerCreator(type, []() -> AlertRulePtr { \
                return std::make_shared<class_name>(); \
            }); \
        } \
    } _alert_rule_registrar_##class_name;

} // namespace rules
} // namespace ai_stream