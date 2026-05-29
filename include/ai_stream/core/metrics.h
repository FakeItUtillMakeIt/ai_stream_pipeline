// include/ai_stream/core/metrics.h
#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

namespace ai_stream {
namespace core {

struct NodeMetrics {
    std::string node_name;
    std::string pipeline_id;

    uint64_t total_packets = 0;
    uint64_t dropped_packets = 0;
    uint64_t total_latency_ms = 0;
    uint64_t min_latency_ms = 0;
    uint64_t max_latency_ms = 0;
    uint64_t last_latency_ms = 0;
    double fps = 0.0;

    uint64_t window_packets = 0;
    std::chrono::steady_clock::time_point window_start;
    std::chrono::steady_clock::time_point last_fps_check;

    NodeMetrics()
        : window_start(std::chrono::steady_clock::now())
        , last_fps_check(std::chrono::steady_clock::now()) {}
};

class MetricsCollector {
public:
    static MetricsCollector& instance();

    void recordLatency(const std::string& pipeline_id, const std::string& node_name, uint64_t latency_ms);
    void recordProcessed(const std::string& pipeline_id, const std::string& node_name);
    void recordDropped(const std::string& pipeline_id, const std::string& node_name);

    void setGpuMemoryMb(int mb);
    void setCpuUsagePercent(int pct);
    void setActivePipelines(int count);
    void setTotalPipelines(int count);

    NodeMetrics getNodeMetrics(const std::string& pipeline_id, const std::string& node_name) const;
    std::vector<NodeMetrics> getPipelineMetrics(const std::string& pipeline_id) const;
    std::unordered_map<std::string, std::vector<NodeMetrics>> getAllMetrics() const;

    std::string formatPrometheus() const;
    std::string formatJson() const;

    void reset(const std::string& pipeline_id);
    void resetAll();

    struct SystemMetrics {
        int gpu_memory_mb = 0;
        int cpu_usage_percent = 0;
        int active_pipelines = 0;
        int total_pipelines = 0;
    };
    SystemMetrics getSystemMetrics() const;

private:
    MetricsCollector() = default;

    struct NodeKey {
        std::string pipeline_id;
        std::string node_name;
        bool operator==(const NodeKey& o) const {
            return pipeline_id == o.pipeline_id && node_name == o.node_name;
        }
    };
    struct NodeKeyHash {
        size_t operator()(const NodeKey& k) const {
            return std::hash<std::string>{}(k.pipeline_id) ^
                   (std::hash<std::string>{}(k.node_name) << 1);
        }
    };

    NodeKey makeKey(const std::string& pipeline_id, const std::string& node_name) const {
        return {pipeline_id, node_name};
    }

    mutable std::mutex mutex_;
    std::unordered_map<NodeKey, NodeMetrics, NodeKeyHash> metrics_;

    std::atomic<int> gpu_memory_mb_{0};
    std::atomic<int> cpu_usage_percent_{0};
    std::atomic<int> active_pipelines_{0};
    std::atomic<int> total_pipelines_{0};
};

} // namespace core
} // namespace ai_stream
