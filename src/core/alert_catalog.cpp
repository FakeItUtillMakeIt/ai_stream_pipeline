// src/core/alert_catalog.cpp
#include "ai_stream/core/alert_catalog.h"

#include <algorithm>
#include <unordered_set>

namespace ai_stream {
namespace core {

namespace {

constexpr const char* kPerson = "ITEM_PERSON_BEHAVIOR";
constexpr const char* kScene = "ITEM_SCENE_RECOGNITION";
constexpr const char* kRelation = "ITEM_RELATION_RECOGNITION";
constexpr const char* kAction = "ITEM_ACTION_RECOGNITION";

std::string normalizeCategory(const std::string& v) {
    if (v.empty()) return v;
    if (v.rfind("ITEM_", 0) == 0) return v;
    if (v == "person" || v == "person_behavior") return kPerson;
    if (v == "scene" || v == "scene_recognition") return kScene;
    if (v == "relation" || v == "relation_recognition") return kRelation;
    if (v == "action" || v == "action_recognition") return kAction;
    return v;
}

std::vector<std::string> stringsOf(const nlohmann::json& obj, const char* key) {
    if (!obj.is_object() || !obj.contains(key)) return {};
    const auto& v = obj[key];
    if (v.is_array()) {
        std::vector<std::string> out;
        for (const auto& e : v) if (e.is_string()) out.push_back(e.get<std::string>());
        return out;
    }
    if (v.is_string()) return {v.get<std::string>()};
    return {};
}

std::vector<std::string> categoriesOf(const nlohmann::json& obj) {
    auto out = stringsOf(obj, "model_categories");
    for (auto& c : out) c = normalizeCategory(c);
    return out;
}

bool contains(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

bool containsAny(const std::vector<std::string>& v, std::initializer_list<const char*> keys) {
    for (const char* k : keys) if (contains(v, k)) return true;
    return false;
}

AlertRuleInfo rule(const char* type, const char* label, const char* category,
                   const char* severity, std::vector<std::string> models = {},
                   std::vector<std::string> classes = {},
                   std::vector<std::string> labels = {}) {
    AlertRuleInfo r;
    r.type = type;
    r.label = label;
    r.category = category;
    r.severity = severity;
    r.required_models = std::move(models);
    r.required_classes = std::move(classes);
    r.required_labels = std::move(labels);
    return r;
}

const std::vector<AlertCategoryInfo>& categoriesStorage() {
    static const std::vector<AlertCategoryInfo> v = {
        {kPerson, "人员行为", 1},
        {kScene, "场景识别", 2},
        {kRelation, "关系识别", 3},
        {kAction, "动作识别", 4},
    };
    return v;
}

const std::vector<AlertRuleInfo>& rulesStorage() {
    static const std::vector<AlertRuleInfo> v = {
        // ===== 人员行为 / 劳保（9cls 模型）=====
        rule("person_intrusion", "人员入侵", kPerson, "P0", {"person_behavior_det"}, {"person"}),
        rule("person_loitering", "人员徘徊", kPerson, "P2", {"person_behavior_det"}, {"person"}),
        rule("absence", "离岗", kPerson, "P1", {"person_behavior_det"}, {"person"}),
        rule("sleep_on_duty", "睡岗", kPerson, "P1", {"person_behavior_det"}, {"person", "sleeping"}),
        rule("smoking", "吸烟", kPerson, "P2", {"person_behavior_det"}, {"person", "smoking"}),
        rule("phone_call", "打电话", kPerson, "P2", {"person_behavior_det"}, {"person", "phone"}),
        rule("moving_phone_call", "移动中打电话", kPerson, "P2", {"person_behavior_det"}, {"person", "phone"}),
        rule("human_gathering", "人员聚集", kPerson, "P2", {"person_behavior_det"}, {"person"}),
        rule("falling", "跌倒(过程)", kPerson, "P0", {"person_behavior_det"}, {"person", "fall_down"}),
        rule("fall_down", "跌倒(检测框)", kPerson, "P0", {"person_behavior_det"}, {"person", "fall_down"}),
        rule("rising", "起身", kPerson, "P2", {"person_behavior_det"}, {"person"}),
        rule("photographer", "揽拍", kPerson, "P2", {"person_behavior_det"}, {"person"}),
        rule("unlicensed_verdor", "无证摊贩", kPerson, "P2", {"person_behavior_det"}, {"person"}),
        rule("missing_helmet", "未戴安全帽", kPerson, "P1", {"person_behavior_det"}, {"person", "head", "helmet"}),
        rule("missing_work_clothes", "未穿工作服", kPerson, "P1", {"person_behavior_det"}, {"person", "work_clothes"}),
        rule("missing_safety_belt", "未系安全带", kPerson, "P1", {"person_behavior_det"}, {"person", "safety_belt"}),

        // ===== 动作识别（9cls + action）=====
        rule("climbing", "攀爬", kAction, "P1", {"person_behavior_det", "action"}, {"person"}, {"climbing"}),
        rule("fighting", "打架", kAction, "P0", {"person_behavior_det", "action"}, {"person"}, {"fighting"}),

        // ===== 场景识别（scene.engine）=====
        rule("discover_crystal", "发现结晶", kScene, "P1", {"scene_det"}, {"crystal"}),
        rule("discover_visible_fire", "发现火焰", kScene, "P0", {"scene_det"}, {"fire"}),
        rule("discover_smoke", "发现烟雾", kScene, "P0", {"scene_det"}, {"smoke"}),
        rule("discover_hose_cutoff", "发现软管断流", kScene, "P1", {"scene_det"}, {"hose"}),
        // 消防通道占用基于参考图相似度，不依赖场景检测模型
        rule("fire_lane_occupancy", "消防通道占用", kScene, "P2"),

        // ===== 关系识别（relation + boundary）=====
        rule("child_near_boundary", "儿童靠近边界", kRelation, "P0",
             {"relation", "boundary_det"}, {"child", "wall", "gate"}),
    };
    return v;
}

} // namespace

const std::vector<AlertCategoryInfo>& alertCategories() { return categoriesStorage(); }
const std::vector<AlertRuleInfo>& alertRules() { return rulesStorage(); }

const AlertRuleInfo* findAlertRule(const std::string& type) {
    for (const auto& r : rulesStorage()) {
        if (r.type == type) return &r;
    }
    return nullptr;
}

nlohmann::json alertCatalogJson() {
    nlohmann::json cats = nlohmann::json::array();
    for (const auto& c : categoriesStorage()) {
        cats.push_back({{"id", c.id}, {"label", c.label}, {"order", c.order}});
    }
    nlohmann::json rules = nlohmann::json::object();
    for (const auto& r : rulesStorage()) {
        rules[r.type] = {
            {"label", r.label},
            {"category", r.category},
            {"severity", r.severity},
            {"required_models", r.required_models},
            {"required_classes", r.required_classes},
            {"required_labels", r.required_labels},
        };
    }
    return nlohmann::json{{"categories", cats}, {"rules", rules}};
}

std::vector<ModelCapability> collectModelCapabilities(const nlohmann::json& graph) {
    std::vector<ModelCapability> out;
    if (!graph.is_object() || !graph.contains("nodes") || !graph["nodes"].is_array()) return out;

    for (const auto& node : graph["nodes"]) {
        if (!node.is_object()) continue;
        const std::string id = node.value("id", "");
        const std::string type = node.value("type", "");
        const bool is_model_node =
            type == "detection_infer" || type == "pose_infer" ||
            type == "relation_recognition" || type == "action_recognition_videomae";
        if (!is_model_node) continue;

        nlohmann::json params = node.value("params", nlohmann::json::object());
        nlohmann::json dc = (params.contains("detector_config") && params["detector_config"].is_object())
                                ? params["detector_config"] : nlohmann::json::object();

        ModelCapability cap;
        cap.node_id = id;
        cap.role = dc.value("model_role", params.value("model_role", std::string()));
        cap.engine = dc.value("model_path", params.value("model_path", std::string()));
        cap.categories = !categoriesOf(dc).empty() ? categoriesOf(dc) : categoriesOf(params);
        cap.classes = stringsOf(dc, "model_class_names");
        if (cap.classes.empty()) cap.classes = stringsOf(params, "model_class_names");
        cap.labels = stringsOf(params, "action_labels");

        if (type == "relation_recognition") {
            if (cap.role.empty()) cap.role = "relation";
            auto boundary = stringsOf(params, "boundary_classes");
            cap.classes.insert(cap.classes.end(), boundary.begin(), boundary.end());
            if (cap.categories.empty()) cap.categories = {kRelation};
        } else if (type == "action_recognition_videomae") {
            if (cap.role.empty()) cap.role = "action";
            if (cap.categories.empty()) cap.categories = {kAction};
        } else if (type == "pose_infer") {
            if (cap.role.empty()) cap.role = "pose";
            if (cap.categories.empty()) cap.categories = {kPerson};
        } else {
            // detection_infer：没有显式声明时按类别名推断模型族
            if (cap.categories.empty()) {
                if (cap.role == "person_behavior_det") cap.categories = {kPerson};
                else if (cap.role == "scene_det") cap.categories = {kScene};
                else if (cap.role == "boundary_det" || cap.role == "relation") cap.categories = {kRelation};
                else if (cap.role == "action") cap.categories = {kAction};
                else if (containsAny(cap.classes, {"fire", "smoke", "crystal", "hose"}))
                    cap.categories = {kScene};
                else if (containsAny(cap.classes, {"person", "head", "helmet", "work_clothes", "safety_belt", "smoking", "phone", "sleeping", "fall_down"}))
                    cap.categories = {kPerson};
                else if (containsAny(cap.classes, {"child", "wall", "gate"}))
                    cap.categories = {kRelation};
            }
            if (cap.role.empty()) {
                if (contains(cap.categories, kScene)) cap.role = "scene_det";
                else if (contains(cap.categories, kPerson)) cap.role = "person_behavior_det";
                else if (contains(cap.categories, kRelation)) cap.role = "boundary_det";
                else if (contains(cap.categories, kAction)) cap.role = "action";
            }
        }

        if (cap.role.empty() && cap.categories.empty()) continue;
        out.push_back(std::move(cap));
    }
    return out;
}

AlertBindingValidation validateAlertBindings(const nlohmann::json& graph) {
    AlertBindingValidation result;
    if (!graph.is_object() || !graph.contains("nodes") || !graph["nodes"].is_array()) return result;

    const auto caps = collectModelCapabilities(graph);
    auto hasCategory = [&](const std::string& category) {
        for (const auto& c : caps) if (contains(c.categories, category)) return true;
        return false;
    };
    auto hasRole = [&](const std::string& role) {
        for (const auto& c : caps) if (c.role == role) return true;
        return false;
    };
    auto hasClassInRoles = [&](const std::vector<std::string>& roles,
                               const std::string& cls) {
        for (const auto& c : caps) {
            if (!contains(roles, c.role)) continue;
            if (contains(c.classes, cls)) return true;
        }
        return false;
    };
    auto hasLabelInRoles = [&](const std::vector<std::string>& roles,
                               const std::string& label) {
        for (const auto& c : caps) {
            if (!contains(roles, c.role)) continue;
            if (contains(c.labels, label)) return true;
        }
        return false;
    };

    std::unordered_set<std::string> seen;
    for (const auto& node : graph["nodes"]) {
        if (!node.is_object() || node.value("type", "") != "alert") continue;
        const nlohmann::json params = node.value("params", nlohmann::json::object());
        if (!params.contains("rules") || !params["rules"].is_array()) continue;
        for (const auto& r : params["rules"]) {
            if (!r.is_object()) continue;
            const std::string type = r.value("type", "");
            if (type.empty() || !seen.insert(type).second) continue;

            const AlertRuleInfo* meta = findAlertRule(type);
            if (!meta) {
                result.warnings.push_back("告警规则 " + type + " 未加入告警目录，跳过模型校验");
                continue;
            }
            if (meta->required_models.empty()) continue;

            if (!hasCategory(meta->category)) {
                result.errors.push_back("规则 " + meta->label + "(" + type + ") 需要场景类别 " +
                                        meta->category + "，当前模板没有对应模型");
            }
            for (const auto& role : meta->required_models) {
                if (!hasRole(role)) {
                    result.errors.push_back("规则 " + meta->label + "(" + type + ") 需要模型角色 " + role);
                }
            }
            for (const auto& cls : meta->required_classes) {
                if (!hasClassInRoles(meta->required_models, cls)) {
                    result.errors.push_back("规则 " + meta->label + "(" + type + ") 需要模型类别 " + cls);
                }
            }
            for (const auto& label : meta->required_labels) {
                if (!hasLabelInRoles(meta->required_models, label)) {
                    result.errors.push_back("规则 " + meta->label + "(" + type + ") 需要动作标签 " + label);
                }
            }
        }
    }

    result.ok = result.errors.empty();
    return result;
}

} // namespace core
} // namespace ai_stream
