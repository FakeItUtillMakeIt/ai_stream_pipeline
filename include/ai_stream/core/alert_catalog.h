// include/ai_stream/core/alert_catalog.h
// 告警场景目录与模型依赖校验。
// - 分类用于研发/实施页面按场景分组；
// - 依赖用于构建前校验：规则启用了，但模板里没有对应模型能力时直接拒绝。
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace ai_stream {
namespace core {

struct AlertCategoryInfo {
    std::string id;       // ITEM_PERSON_BEHAVIOR / ...
    std::string label;    // 中文名
    int order = 0;
};

struct AlertRuleInfo {
    std::string type;       // 规则工厂 key，如 person_intrusion
    std::string label;      // 中文名
    std::string category;   // AlertItemType id
    std::string severity;   // P0 / P1 / P2
    std::vector<std::string> required_models;   // 模型角色
    std::vector<std::string> required_classes;  // 模型类别
    std::vector<std::string> required_labels;   // 动作标签
};

const std::vector<AlertCategoryInfo>& alertCategories();
const std::vector<AlertRuleInfo>& alertRules();
const AlertRuleInfo* findAlertRule(const std::string& type);
nlohmann::json alertCatalogJson();

struct ModelCapability {
    std::string node_id;
    std::string role;
    std::string engine;
    std::vector<std::string> categories;
    std::vector<std::string> classes;
    std::vector<std::string> labels;
};

// 从 pipeline graph JSON 中提取模型能力（显式声明，或按类别名推断）。
std::vector<ModelCapability> collectModelCapabilities(const nlohmann::json& graph);

struct AlertBindingValidation {
    bool ok = true;
    std::vector<std::string> errors;
    std::vector<std::string> warnings;
};

// 校验 alert 节点里启用的规则是否有对应模型/类别/标签。
AlertBindingValidation validateAlertBindings(const nlohmann::json& graph);

} // namespace core
} // namespace ai_stream
