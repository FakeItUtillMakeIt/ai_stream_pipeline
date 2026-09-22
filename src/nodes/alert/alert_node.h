// src/nodes/alert/alert_node.h
#pragma once

#include "ai_stream/core/queued_node.h"
#include "ai_stream/rules/i_alert_rule.h"
#include "3rd_party/thread_pool/thread_pool.hpp"
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include <memory>
#include <filesystem>

namespace ai_stream {
namespace nodes {

using AlertCallback = std::function<void(const rules::AlertEvent&)>;

class AlertNode : public core::QueuedNode<core::Node> {
public:
    AlertNode();
    virtual ~AlertNode();

    // QueuedNode 接口
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;
    bool onStartup() override;
    void onShutdown() override;
    bool configureImpl(const std::string& node_id, const nlohmann::json& params) override;

    void addRule(rules::AlertRulePtr rule);
    void setAlertCallback(AlertCallback callback);
    void setSnapshotDir(const std::string& dir);
    nlohmann::json getStatistics() const;

private:
    std::vector<rules::AlertResult> process_all_alerts_parallel(std::shared_ptr<core::InferenceResultPacket> packet);
    std::vector<rules::AlertResult> process_all_alerts_sequence(std::shared_ptr<core::InferenceResultPacket> packet);
    rules::AlertResult process_single_alert(rules::AlertRulePtr rule,
                                    std::shared_ptr<core::InferenceResultPacket> packet);
    
    // 常驻规则执行线程池（替代每帧每规则 std::async，避免线程爆炸）
    void startPool();
    void stopPool();

private:
    std::atomic<bool> enable_parallel_{true};
    std::vector<rules::AlertRulePtr> rules_;
    AlertCallback callback_;
    mutable std::mutex mutex_;
    std::string snapshot_dir_ = "./alerts";

    std::unique_ptr<ThreadPool> pool_;
};

} // namespace nodes
} // namespace ai_stream