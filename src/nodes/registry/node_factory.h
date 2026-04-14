// src/nodes/registry/node_factory.h
#pragma once

#include "ai_stream/core/node.h"
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp>

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
    }
    
    std::shared_ptr<Node> create(const std::string& type, const nlohmann::json& params) {
        auto it = creators_.find(type);
        if (it != creators_.end()) {
            return it->second(params);
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
