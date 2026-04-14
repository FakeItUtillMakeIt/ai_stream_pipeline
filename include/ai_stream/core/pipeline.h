// include/ai_stream/core/pipeline.h
#pragma once

#include "node.h"
#include <memory>
#include <string>
#include <unordered_map>
#include <nlohmann/json.hpp> // 前向声明，避免引入完整 JSON 头文件

namespace ai_stream {
namespace core {

// 管道类：负责根据配置构建节点图，并管理节点的生命周期
class Pipeline : public std::enable_shared_from_this<Pipeline> {
public:
    using NodeMap = std::unordered_map<std::string, std::shared_ptr<Node>>;

    Pipeline(const std::string& id) : id_(id) {}
    ~Pipeline();

    // 从 JSON 配置构建管道
    bool buildFromJson(const nlohmann::json& config);
    
    // 启动/停止管道
    bool start();
    void stop();
    
    // 获取节点 (用于调试或手动干预)
    std::shared_ptr<Node> getNode(const std::string& name);
    
    // 获取管道 ID
    const std::string& getId() const { return id_; }
    
    // 获取管道状态
    bool isRunning() const { return running_; }

private:
    std::string id_;
    NodeMap nodes_;
    std::atomic<bool> running_{false};
    
    // 内部方法：根据类型字符串创建节点实例
    std::shared_ptr<Node> createNodeInstance(const std::string& type, const nlohmann::json& params);
};

} // namespace core
} // namespace ai_stream