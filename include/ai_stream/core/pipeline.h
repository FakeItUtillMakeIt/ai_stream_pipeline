// include/ai_stream/core/pipeline.h
#pragma once

#include "node.h"
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
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
    bool isRunning() const;

private:
    mutable std::mutex mutex_;
    std::string id_;
    NodeMap nodes_;
    std::atomic<bool> running_{false};

    // 拓扑序（上游 -> 下游），用于确定启停顺序
    std::vector<std::string> topo_order_;

    // 内部方法：Kahn 算法计算拓扑序，图中存在环时返回 false
    bool computeTopologicalOrder(const std::vector<std::pair<std::string, std::string>>& edges);
};

} // namespace core
} // namespace ai_stream