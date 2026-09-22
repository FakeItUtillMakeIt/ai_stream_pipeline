// src/core/metrics.cpp
#include "ai_stream/core/metrics.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <sstream>
#include <iomanip>
#include <nlohmann/json.hpp>

namespace ai_stream {
namespace core {

namespace {
// Prometheus 文本格式的 label 转义（防止 pipeline/node 名注入伪造指标）
std::string escapePromLabel(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"':  out += "\\\""; break;
            case '\n': out += "\\n"; break;
            default:   out += c; break;
        }
    }
    return out;
}
} // namespace

MetricsCollector& MetricsCollector::instance() {
    static MetricsCollector collector;
    return collector;
}

void MetricsCollector::record(const std::string& pipeline_id, const std::string& node_name, uint64_t latency_ms) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto& m = metrics_[makeKey(pipeline_id, node_name)];
    m.pipeline_id = pipeline_id;
    m.node_name = node_name;

    m.total_packets++;
    m.window_packets++;
    m.last_latency_ms = latency_ms;
    m.total_latency_ms += latency_ms;
    if (latency_ms < m.min_latency_ms) m.min_latency_ms = latency_ms;
    if (latency_ms > m.max_latency_ms) m.max_latency_ms = latency_ms;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - m.last_fps_check).count();
    if (elapsed >= 1000) {
        m.fps = m.window_packets / (elapsed / 1000.0);
        m.window_packets = 0;
        m.last_fps_check = now;
    }
}

void MetricsCollector::recordLatency(const std::string& pipeline_id, const std::string& node_name, uint64_t latency_ms) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto& m = metrics_[makeKey(pipeline_id, node_name)];
    m.pipeline_id = pipeline_id;
    m.node_name = node_name;

    m.last_latency_ms = latency_ms;
    m.total_latency_ms += latency_ms;

    if (latency_ms < m.min_latency_ms) {
        m.min_latency_ms = latency_ms;
    }
    if (latency_ms > m.max_latency_ms) {
        m.max_latency_ms = latency_ms;
    }
}

void MetricsCollector::recordProcessed(const std::string& pipeline_id, const std::string& node_name) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto& m = metrics_[makeKey(pipeline_id, node_name)];
    m.pipeline_id = pipeline_id;
    m.node_name = node_name;

    m.total_packets++;
    m.window_packets++;

    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        now - m.last_fps_check).count();
    if (elapsed >= 1000) {
        double seconds = elapsed / 1000.0;
        m.fps = m.window_packets / seconds;
        m.window_packets = 0;
        m.last_fps_check = now;
    }
}

void MetricsCollector::recordDropped(const std::string& pipeline_id, const std::string& node_name) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    auto& m = metrics_[makeKey(pipeline_id, node_name)];
    m.pipeline_id = pipeline_id;
    m.node_name = node_name;
    m.dropped_packets++;
}

void MetricsCollector::setGpuMemoryMb(int mb) {
    gpu_memory_mb_.store(mb);
}

void MetricsCollector::setCpuUsagePercent(int pct) {
    cpu_usage_percent_.store(pct);
}

void MetricsCollector::setActivePipelines(int count) {
    active_pipelines_.store(count);
}

void MetricsCollector::setTotalPipelines(int count) {
    total_pipelines_.store(count);
}

NodeMetrics MetricsCollector::getNodeMetrics(const std::string& pipeline_id, const std::string& node_name) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    auto it = metrics_.find(makeKey(pipeline_id, node_name));
    if (it != metrics_.end()) {
        return it->second;
    }
    NodeMetrics empty;
    empty.pipeline_id = pipeline_id;
    empty.node_name = node_name;
    return empty;
}

std::vector<NodeMetrics> MetricsCollector::getPipelineMetrics(const std::string& pipeline_id) const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::vector<NodeMetrics> result;
    for (const auto& [key, m] : metrics_) {
        if (key.pipeline_id == pipeline_id) {
            result.push_back(m);
        }
    }
    return result;
}

std::unordered_map<std::string, std::vector<NodeMetrics>> MetricsCollector::getAllMetrics() const {
    std::shared_lock<std::shared_mutex> lock(mutex_);
    std::unordered_map<std::string, std::vector<NodeMetrics>> result;
    for (const auto& [key, m] : metrics_) {
        result[key.pipeline_id].push_back(m);
    }
    return result;
}

MetricsCollector::SystemMetrics MetricsCollector::getSystemMetrics() const {
    SystemMetrics s;
    s.gpu_memory_mb = gpu_memory_mb_.load();
    s.cpu_usage_percent = cpu_usage_percent_.load();
    s.active_pipelines = active_pipelines_.load();
    s.total_pipelines = total_pipelines_.load();
    return s;
}

void MetricsCollector::reset(const std::string& pipeline_id) {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    for (auto it = metrics_.begin(); it != metrics_.end(); ) {
        if (it->first.pipeline_id == pipeline_id) {
            it = metrics_.erase(it);
        } else {
            ++it;
        }
    }
}

void MetricsCollector::resetAll() {
    std::unique_lock<std::shared_mutex> lock(mutex_);
    metrics_.clear();
}

std::string MetricsCollector::formatJson() const {
    nlohmann::json j;
    auto all = getAllMetrics();
    auto sys = getSystemMetrics();

    nlohmann::json pipelines_json = nlohmann::json::object();
    for (const auto& [pid, nodes] : all) {
        nlohmann::json nodes_json = nlohmann::json::array();
        for (const auto& m : nodes) {
            const double avg = m.total_packets > 0
                ? static_cast<double>(m.total_latency_ms) / m.total_packets : 0.0;
            const uint64_t minv = m.total_packets > 0 ? m.min_latency_ms : 0;
            nodes_json.push_back({
                {"node_name", m.node_name},
                {"total_packets", m.total_packets},
                {"dropped_packets", m.dropped_packets},
                {"latency_ms", {
                    {"avg", avg},
                    {"min", minv},
                    {"max", m.max_latency_ms},
                    {"last", m.last_latency_ms}
                }},
                {"fps", m.fps}
            });
        }
        pipelines_json[pid] = nodes_json;
    }

    j["pipelines"] = pipelines_json;
    j["system"] = {
        {"gpu_memory_mb", sys.gpu_memory_mb},
        {"cpu_percent", sys.cpu_usage_percent},
        {"active_pipelines", sys.active_pipelines},
        {"total_pipelines", sys.total_pipelines}
    };

    return j.dump(2);
}

std::string MetricsCollector::formatPrometheus() const {
    std::ostringstream oss;
    auto all = getAllMetrics();
    auto sys = getSystemMetrics();

    oss << "# HELP ai_stream_node_packets_total Total packets processed by node\n";
    oss << "# TYPE ai_stream_node_packets_total counter\n";
    for (const auto& [pid, nodes] : all) {
        const std::string epid = escapePromLabel(pid);
        for (const auto& m : nodes) {
            oss << "ai_stream_node_packets_total{pipeline=\"" << epid
                << "\",node=\"" << escapePromLabel(m.node_name) << "\"} "
                << m.total_packets << "\n";
        }
    }

    oss << "\n# HELP ai_stream_node_packets_dropped Total packets dropped by node\n";
    oss << "# TYPE ai_stream_node_packets_dropped counter\n";
    for (const auto& [pid, nodes] : all) {
        const std::string epid = escapePromLabel(pid);
        for (const auto& m : nodes) {
            oss << "ai_stream_node_packets_dropped{pipeline=\"" << epid
                << "\",node=\"" << escapePromLabel(m.node_name) << "\"} "
                << m.dropped_packets << "\n";
        }
    }

    oss << "\n# HELP ai_stream_node_latency_ms Node processing latency in ms\n";
    oss << "# TYPE ai_stream_node_latency_ms gauge\n";
    for (const auto& [pid, nodes] : all) {
        const std::string epid = escapePromLabel(pid);
        for (const auto& m : nodes) {
            const double avg = m.total_packets > 0
                ? static_cast<double>(m.total_latency_ms) / m.total_packets : 0.0;
            const uint64_t minv = m.total_packets > 0 ? m.min_latency_ms : 0;
            const std::string enode = escapePromLabel(m.node_name);
            oss << "ai_stream_node_latency_ms{pipeline=\"" << epid
                << "\",node=\"" << enode << "\",type=\"avg\"} "
                << avg << "\n";
            oss << "ai_stream_node_latency_ms{pipeline=\"" << epid
                << "\",node=\"" << enode << "\",type=\"min\"} "
                << minv << "\n";
            oss << "ai_stream_node_latency_ms{pipeline=\"" << epid
                << "\",node=\"" << enode << "\",type=\"max\"} "
                << m.max_latency_ms << "\n";
            oss << "ai_stream_node_latency_ms{pipeline=\"" << epid
                << "\",node=\"" << enode << "\",type=\"last\"} "
                << m.last_latency_ms << "\n";
        }
    }

    oss << "\n# HELP ai_stream_node_fps Current node throughput in fps\n";
    oss << "# TYPE ai_stream_node_fps gauge\n";
    for (const auto& [pid, nodes] : all) {
        const std::string epos = escapePromLabel(pid);
        for (const auto& m : nodes) {
            oss << "ai_stream_node_fps{pipeline=\"" << epos
                << "\",node=\"" << escapePromLabel(m.node_name) << "\"} "
                << std::fixed << std::setprecision(2) << m.fps << "\n";
        }
    }

    oss << "\n# HELP ai_stream_system_gpu_memory_mb GPU memory usage in MB\n";
    oss << "# TYPE ai_stream_system_gpu_memory_mb gauge\n";
    oss << "ai_stream_system_gpu_memory_mb " << sys.gpu_memory_mb << "\n";

    oss << "\n# HELP ai_stream_system_cpu_percent CPU usage percent\n";
    oss << "# TYPE ai_stream_system_cpu_percent gauge\n";
    oss << "ai_stream_system_cpu_percent " << sys.cpu_usage_percent << "\n";

    oss << "\n# HELP ai_stream_system_active_pipelines Active pipeline count\n";
    oss << "# TYPE ai_stream_system_active_pipelines gauge\n";
    oss << "ai_stream_system_active_pipelines " << sys.active_pipelines << "\n";

    oss << "\n# HELP ai_stream_system_total_pipelines Total pipeline count\n";
    oss << "# TYPE ai_stream_system_total_pipelines gauge\n";
    oss << "ai_stream_system_total_pipelines " << sys.total_pipelines << "\n";

    return oss.str();
}

} // namespace core
} // namespace ai_stream
