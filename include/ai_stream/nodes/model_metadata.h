// include/ai_stream/nodes/model_metadata.h
// 推理节点的模型能力声明：模型角色 + 支持的告警场景类别。
// 仅做配置解析与查询，不依赖推理后端。
#pragma once

#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace ai_stream {
namespace nodes {

struct ModelMetadata {
    std::string role;                       // person_behavior_det / scene_det / relation / action / boundary_det
    std::vector<std::string> categories;    // ITEM_PERSON_BEHAVIOR / ...

    void parse(const nlohmann::json& params) {
        auto read = [this](const nlohmann::json& obj) {
            if (!obj.is_object()) return;
            if (obj.contains("model_role") && obj["model_role"].is_string()) {
                role = obj["model_role"].get<std::string>();
            }
            if (obj.contains("model_categories")) {
                if (obj["model_categories"].is_array()) {
                    categories = obj["model_categories"].get<std::vector<std::string>>();
                } else if (obj["model_categories"].is_string()) {
                    categories = {obj["model_categories"].get<std::string>()};
                }
            }
        };
        read(params);
        if (params.contains("detector_config") && params["detector_config"].is_object()) {
            read(params["detector_config"]);
        }
    }

    nlohmann::json toJson() const {
        return nlohmann::json{{"model_role", role}, {"model_categories", categories}};
    }
};

} // namespace nodes
} // namespace ai_stream
