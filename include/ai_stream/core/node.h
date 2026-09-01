// include/ai_stream/core/node.h
#pragma once

#include "utils/time_util.h"
#include "packet.h"
#include "ai_stream/core/metrics.h"
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
    virtual bool isRunning() const {return running_.load();}
    
    // 核心数据处理入口
    virtual void pushData(std::shared_ptr<BasePacket> packet) = 0;

    // 下游能力：默认只接受 CPU packet。GPU 友好节点可覆写为 true。
    virtual bool acceptsGpuFrame() const { return false; }

    // 从 JSON 参数配置节点（由 Pipeline::buildFromJson 调用）
    // 返回 false 表示配置无效或初始化失败（如模型加载失败），管道构建将终止
    virtual bool configure(const std::string& node_id, const nlohmann::json& params) {
        (void)node_id;
        (void)params;
        return true;
    }

    // 节点名称
    const std::string& getName() const { return name_; }

    // 设置节点名称（Pipeline 构建时设为配置中的节点 id）
    void setName(const std::string& name) { name_ = name; }
    
    // 添加下游节点 (一流多用的关键)
    void addDownstream(std::shared_ptr<Node> downstream) {
        if (downstream) {
            downstreams_.push_back(downstream);
        }
    }
    
    // 设置管道上下文 (用于获取全局配置等)
    void setPipeline(std::weak_ptr<Pipeline> pipeline) { pipeline_ = pipeline; }

    // 记录节点处理指标（节点在 pushData 中计算完 latency 后调用）
    void recordMetrics(uint64_t latency_ms) {
        recordMetricsImpl(latency_ms, false);
    }
    void recordDropped() {
        recordMetricsImpl(0, true);
    }

protected:
    // 广播数据给所有下游节点，并自动记录当前节点的处理指标
    void broadcast(std::shared_ptr<BasePacket> packet);

    std::string name_;
    std::vector<std::weak_ptr<Node>> downstreams_;
    std::weak_ptr<Pipeline> pipeline_;
    std::atomic<bool> running_{false};
    uint64_t in_time_ms_ = 0;

private:
    void recordMetricsImpl(uint64_t latency_ms, bool dropped);
};

// 为了方便类型转换，提供一个辅助宏
#define NODE_TYPE_CAST(T, node_ptr) std::dynamic_pointer_cast<T>(node_ptr)

} // namespace core
} // namespace ai_stream
