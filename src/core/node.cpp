// src/core/node.cpp
#include "ai_stream/core/node.h"
#include "ai_stream/core/pipeline.h" // 完整定义
#include <algorithm>

namespace ai_stream {
namespace core {

void Node::broadcast(std::shared_ptr<BasePacket> packet) {
    auto it = packet->cost_time_map.find(name_);
    if (it != packet->cost_time_map.end()) {
        recordMetricsImpl(it->second, false);
    }
    bool has_expired = false;
    for (auto& weak_down : downstreams_) {
        if (auto down = weak_down.lock()) {
            down->pushData(packet);
        } else {
            has_expired = true;
        }
    }
    // 清理已失效的下游弱引用（构建完成后 downstreams_ 无并发写入）
    if (has_expired) {
        downstreams_.erase(
            std::remove_if(downstreams_.begin(), downstreams_.end(),
                           [](const std::weak_ptr<Node>& w) { return w.expired(); }),
            downstreams_.end());
    }
}

void Node::recordMetricsImpl(uint64_t latency_ms, bool dropped) {
    std::string pid;
    if (auto p = pipeline_.lock()) {
        pid = p->getId();
    }
    auto& mc = MetricsCollector::instance();
    if (dropped) {
        mc.recordDropped(pid, name_);
    } else {
        mc.recordLatency(pid, name_, latency_ms);
        mc.recordProcessed(pid, name_);
    }
}

} // namespace core
} // namespace ai_stream