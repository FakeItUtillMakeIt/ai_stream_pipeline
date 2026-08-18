// src/rules/alert/alert_rule_factory.h
#pragma once

#include "ai_stream/rules/i_alert_rule.h"
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <iostream>
#include "3rd_party/log_mgr/log_mgr.h"

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
        // 使用与 NodeFactory 相同的日志方式
        if (auto logger = spdlog::get("ai_stream")) {
            LOG_INFO_FMT("[AlertRuleFactory] Registered alert rule type: {}", type);
            LOG_INFO_FMT("[AlertRuleFactory] Total registered rules: {}", creators_.size());
        } else {
            // 备用：如果没有 logger，输出到控制台
            std::cout << "[AlertRuleFactory] Registered alert rule type: " << type << std::endl;
        }
    }

    AlertRulePtr create(const std::string& type) {
        if (auto logger = spdlog::get("ai_stream")) {
            LOG_INFO_FMT("[AlertRuleFactory] Creating rule type: {}", type);
            LOG_INFO_FMT("[AlertRuleFactory] Registered rules count: {}", creators_.size());
            
            // 打印所有已注册的规则类型
            for (const auto& [key, value] : creators_) {
                LOG_INFO_FMT("[AlertRuleFactory]   Available: {}", key);
            }
        } else {
            std::cout << "[AlertRuleFactory] Creating rule type: " << type << std::endl;
            std::cout << "[AlertRuleFactory] Registered rules count: " << creators_.size() << std::endl;
            for (const auto& [key, value] : creators_) {
                std::cout << "[AlertRuleFactory]   Available: " << key << std::endl;
            }
        }

        auto it = creators_.find(type);
        if (it != creators_.end()) {
            return it->second();
        }

        if (auto logger = spdlog::get("ai_stream")) {
            LOG_ERROR_FMT("[AlertRuleFactory] Alert rule type not found: {}", type);
        } else {
            std::cout << "[AlertRuleFactory] Alert rule type not found: " << type << std::endl;
        }
        return nullptr;
    }

private:
    AlertRuleFactory() = default;
    std::map<std::string, AlertRuleCreator> creators_;
};

} // namespace rules
} // namespace ai_stream

// 变量名按行号拼接，支持同一 .cpp 文件内注册多个规则（含别名）
#define REGISTER_ALERT_RULE(type, class_name) REGISTER_ALERT_RULE_IMPL(type, class_name, __LINE__)
#define REGISTER_ALERT_RULE_IMPL(type, class_name, line) REGISTER_ALERT_RULE_IMPL2(type, class_name, line)
#define REGISTER_ALERT_RULE_IMPL2(type, class_name, line) \
    namespace { \
        struct AlertRuleRegistrar_##line { \
            AlertRuleRegistrar_##line() { \
                ai_stream::rules::AlertRuleFactory::instance().registerCreator( \
                    type, []() -> ai_stream::rules::AlertRulePtr { \
                        return std::make_shared<class_name>(); \
                    }); \
            } \
        }; \
        static AlertRuleRegistrar_##line _alert_rule_registrar_##line; \
    }