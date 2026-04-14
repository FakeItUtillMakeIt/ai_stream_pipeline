// include/ai_stream/core/node.h
#pragma once

#include "packet.h"
#include <atomic>
#include <memory>
#include <string>
#include <vector>
#include <functional>

namespace ai_stream {
namespace core {

// 前向声明
class Pipeline;

// 基础节点抽象类 (推模式)
class Node : public std::enable_shared_from_this<Node> {
public:
    explicit Node(const std::string& name = "UnnamedNode") : name_(name) {}
    virtual ~Node() = default;

    // 生命周期管理
    virtual bool start() = 0;
    virtual void stop() = 0;
    
    // 核心数据处理入口
    virtual void pushData(std::shared_ptr<BasePacket> packet) = 0;
    
    // 节点名称
    const std::string& getName() const { return name_; }
    
    // 添加下游节点 (一流多用的关键)
    void addDownstream(std::shared_ptr<Node> downstream) {
        if (downstream) {
            downstreams_.push_back(downstream);
        }
    }
    
    // 设置管道上下文 (用于获取全局配置等)
    void setPipeline(std::weak_ptr<Pipeline> pipeline) { pipeline_ = pipeline; }

protected:
    // 广播数据给所有下游节点
    void broadcast(std::shared_ptr<BasePacket> packet) {
        for (auto& weak_down : downstreams_) {
            if (auto down = weak_down.lock()) {
                down->pushData(packet);
            }
        }
    }
    
    std::string name_;
    std::vector<std::weak_ptr<Node>> downstreams_;
    std::weak_ptr<Pipeline> pipeline_;
    std::atomic<bool> running_{false};
};

// 为了方便类型转换，提供一个辅助宏
#define NODE_TYPE_CAST(T, node_ptr) std::dynamic_pointer_cast<T>(node_ptr)

} // namespace core
} // namespace ai_stream