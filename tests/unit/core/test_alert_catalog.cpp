// 告警目录与模型绑定校验回归测试
#include <gtest/gtest.h>
#include "ai_stream/core/alert_catalog.h"

using nlohmann::json;
using namespace ai_stream::core;

namespace {

json modelNode(const std::string& id, const std::string& type, json params) {
    return json{{"id", id}, {"type", type}, {"params", std::move(params)}};
}

json alertNode(std::initializer_list<const char*> rules) {
    json arr = json::array();
    for (const char* r : rules) arr.push_back({{"type", r}, {"params", {{"name", r}}}});
    return json{{"id", "alert1"}, {"type", "alert"}, {"params", {{"rules", arr}}}};
}

json personModel() {
    return modelNode("person", "detection_infer", {
        {"detector_config", {
            {"model_role", "person_behavior_det"},
            {"model_categories", json::array({"ITEM_PERSON_BEHAVIOR"})},
            {"model_class_names", json::array({"person", "head", "helmet", "work_clothes",
                                               "phone", "smoking", "fall_down", "safety_belt", "sleeping"})}
        }}
    });
}

json actionModel() {
    return modelNode("action", "action_recognition_videomae", {
        {"model_role", "action"},
        {"model_categories", json::array({"ITEM_ACTION_RECOGNITION"})},
        {"action_labels", json::array({"fighting", "climbing", "other"})}
    });
}

json sceneModel() {
    return modelNode("scene", "detection_infer", {
        {"detector_config", {
            {"model_role", "scene_det"},
            {"model_categories", json::array({"ITEM_SCENE_RECOGNITION"})},
            {"model_class_names", json::array({"crystal", "fire", "smoke", "hose"})}
        }}
    });
}

json relationModels() {
    return json::array({
        modelNode("rel", "relation_recognition", {
            {"model_role", "relation"},
            {"model_categories", json::array({"ITEM_RELATION_RECOGNITION"})},
            {"boundary_classes", json::array({"wall", "gate"})}
        }),
        modelNode("boundary", "detection_infer", {
            {"detector_config", {
                {"model_role", "boundary_det"},
                {"model_categories", json::array({"ITEM_RELATION_RECOGNITION"})},
                {"model_class_names", json::array({"child", "wall", "gate"})}
            }}
        }),
    });
}

json graphWith(const json& models, const json& alert) {
    json nodes = json::array();
    if (models.is_array()) for (const auto& m : models) nodes.push_back(m);
    else nodes.push_back(models);
    nodes.push_back(alert);
    return json{{"nodes", nodes}, {"edges", json::array()}};
}

} // namespace

TEST(AlertCatalogTest, FightingRequiresPersonAndAction) {
    auto ok = validateAlertBindings(graphWith(json::array({personModel(), actionModel()}), alertNode({"fighting"})));
    EXPECT_TRUE(ok.ok) << (ok.errors.empty() ? "" : ok.errors.front());

    auto missingAction = validateAlertBindings(graphWith(personModel(), alertNode({"fighting"})));
    EXPECT_FALSE(missingAction.ok);
}

TEST(AlertCatalogTest, SceneRulesRequireSceneModel) {
    auto missing = validateAlertBindings(graphWith(personModel(), alertNode({"discover_smoke"})));
    EXPECT_FALSE(missing.ok);

    auto ok = validateAlertBindings(graphWith(sceneModel(), alertNode({"discover_smoke"})));
    EXPECT_TRUE(ok.ok) << (ok.errors.empty() ? "" : ok.errors.front());
}

TEST(AlertCatalogTest, RelationRuleRequiresBothModels) {
    auto ok = validateAlertBindings(graphWith(relationModels(), alertNode({"child_near_boundary"})));
    EXPECT_TRUE(ok.ok) << (ok.errors.empty() ? "" : ok.errors.front());

    auto missingBoundary = validateAlertBindings(graphWith(
        modelNode("rel", "relation_recognition", {
            {"model_role", "relation"},
            {"model_categories", json::array({"ITEM_RELATION_RECOGNITION"})},
            {"boundary_classes", json::array({"wall", "gate"})}
        }), alertNode({"child_near_boundary"})));
    EXPECT_FALSE(missingBoundary.ok);
}

TEST(AlertCatalogTest, CatalogCoversEveryRule) {
    for (const auto& r : alertRules()) {
        EXPECT_NE(findAlertRule(r.type), nullptr) << r.type;
        EXPECT_FALSE(r.category.empty()) << r.type;
    }
    auto jsonCatalog = alertCatalogJson();
    EXPECT_EQ(jsonCatalog["categories"].size(), alertCategories().size());
    EXPECT_EQ(jsonCatalog["rules"].size(), alertRules().size());
}
