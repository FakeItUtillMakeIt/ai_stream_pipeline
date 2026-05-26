// src/nodes/alert/alert_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "ai_stream/rules/i_alert_rule.h"
#include <vector>
#include <functional>
#include <mutex>
#include <atomic>
#include <filesystem>

namespace ai_stream {
namespace nodes {

using AlertCallback = std::function<void(const rules::AlertEvent&)>;

class AlertNode : public core::Node {
public:
    AlertNode();
    virtual ~AlertNode();

    bool start() override;
    void stop() override;
    void pushData(std::shared_ptr<core::BasePacket> packet) override;
    bool isRunning() const override{return running_.load();}

    void addRule(rules::AlertRulePtr rule);
    void setAlertCallback(AlertCallback callback);
    void setSnapshotDir(const std::string& dir);
    nlohmann::json getStatistics() const;

private:
    std::vector<rules::AlertResult> process_all_alerts_parallel(std::shared_ptr<core::InferenceResultPacket> packet);
    std::vector<rules::AlertResult> process_all_alerts_sequence(std::shared_ptr<core::InferenceResultPacket> packet);
    rules::AlertResult process_single_alert(rules::AlertRulePtr rule,
                                    std::shared_ptr<core::InferenceResultPacket> packet);
    
    void handleEvents(const std::vector<rules::AlertEvent>& events);
    void saveSnapshot(std::shared_ptr<core::InferenceResultPacket> packet,
                      const rules::AlertEvent& event);

private:
    std::atomic<bool> enable_parallel_{true};
    std::vector<rules::AlertRulePtr> rules_;
    AlertCallback callback_;
    mutable std::mutex mutex_;
    std::atomic<bool> running_{false};
    std::string snapshot_dir_ = "./alerts";
};

} // namespace nodes
} // namespace ai_stream