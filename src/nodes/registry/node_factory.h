// src/nodes/registry/node_factory.h
#pragma once

#include "ai_stream/core/node.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace core {

using NodeCreator = std::function<std::shared_ptr<Node>(const nlohmann::json&)>;

class NodeFactory {
public:
    static NodeFactory& instance() {
        static NodeFactory factory;
        return factory;
    }

    void registerCreator(const std::string& type, NodeCreator creator) {
        creators_[type] = std::move(creator);
        if (auto logger = spdlog::get("ai_stream")) {
            LOG_INFO_FMT("[NodeFactory] Registered node type: {}", type);
        }
    }

    std::shared_ptr<Node> create(const std::string& type, const nlohmann::json& params) {
        if (auto logger = spdlog::get("ai_stream")) {
            LOG_INFO_FMT("[NodeFactory] Creating node type: {}", type);
            LOG_INFO_FMT("[NodeFactory] Registered node types count: {}", creators_.size());
            for (const auto& [key, value] : creators_) {
                LOG_INFO_FMT("[NodeFactory]   Registered: {}", key);
            }
        }

        auto it = creators_.find(type);
        if (it != creators_.end()) {
            if (auto logger = spdlog::get("ai_stream")) {
                LOG_INFO_FMT("[NodeFactory] Found creator for type: {}", type);
            }
            return it->second(params);
        }

        if (auto logger = spdlog::get("ai_stream")) {
            LOG_ERROR_FMT("[NodeFactory] Node type not found: {}", type);
        }
        return nullptr;
    }

private:
    NodeFactory() = default;
    std::unordered_map<std::string, NodeCreator> creators_;
};

template<typename T>
class NodeRegistrar {
public:
    NodeRegistrar(const std::string& type) {
        NodeFactory::instance().registerCreator(type, [](const nlohmann::json& params) -> std::shared_ptr<Node> {
            return std::make_shared<T>();
        });
    }
};

} // namespace core
} // namespace ai_stream

// 注册宏 - 放在匿名命名空间中，避免符号冲突
// 每个 .cpp 文件只有一个注册，使用固定名称即可
#define REGISTER_NODE(type, full_class_name) \
    namespace { \
        static ai_stream::core::NodeRegistrar<full_class_name> _node_registrar(type); \
    }
