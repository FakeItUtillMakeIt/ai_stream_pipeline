// src/nodes/alert/alert_node.cpp
#include "alert_node.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "rules/alert/alert_rule_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "rules/alert/person_intrusion_rules.h"
#include <opencv2/opencv.hpp>

namespace ai_stream {
namespace nodes {

AlertNode::AlertNode() : core::Node("AlertNode") {
    LOG_INFO("[AlertNode] Constructor");
}

AlertNode::~AlertNode() {
    stop();
    LOG_DEBUG("[AlertNode] Destructor");
}

bool AlertNode::start() {
    running_ = true;
    if (!snapshot_dir_.empty()) {
        std::filesystem::create_directories(snapshot_dir_);
    }
    LOG_INFO_FMT("[AlertNode] Started with {} rules", rules_.size());
    return true;
}

void AlertNode::stop() {
    running_ = false;
    for (auto& r : rules_) r->reset();
    LOG_INFO("[AlertNode] Stopped");
}

void AlertNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    // 透传所有数据到下游
    broadcast(packet);

    if (!running_) return;
    if (packet->type != core::PacketType::META_DATA) return;

    auto infer = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer) return;

    int64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();

    // 处理所有规则
    // todo: 可以考虑并行处理多个规则，提升性能
    for (auto& rule : rules_) {
        auto events = rule->process(infer, now);
        if (!events.empty()) {
            handleEvents(events);
            for (const auto& e : events) {
                if (e.level >= rules::AlertLevel::WARNING) {
                    saveSnapshot(infer, e);
                }
            }
        }
    }
}

void AlertNode::addRule(rules::AlertRulePtr rule) {
    rules_.push_back(std::move(rule));
    LOG_INFO_FMT("[AlertNode] Added rule: {} ({})", rules_.back()->getName(), rules_.back()->getName());
}

void AlertNode::setAlertCallback(AlertCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    callback_ = std::move(callback);
}

void AlertNode::setSnapshotDir(const std::string& dir) {
    snapshot_dir_ = dir;
}

void AlertNode::handleEvents(const std::vector<rules::AlertEvent>& events) {
    for (const auto& e : events) {
        // 日志
        auto level_str = [](rules::AlertLevel l) {
            switch (l) {
                case rules::AlertLevel::INFO: return "INFO";
                case rules::AlertLevel::WARNING: return "WARN";
                case rules::AlertLevel::ERROR: return "ERROR";
                case rules::AlertLevel::CRITICAL: return "CRIT";
            }
            return "UNKNOWN";
        };
        LOG_INFO_FMT("[AlertNode] [{}] {}: {}", level_str(e.level), e.rule_name, e.description);

        // 回调
        std::lock_guard<std::mutex> lock(mutex_);
        if (callback_) callback_(e);
    }
}

void AlertNode::saveSnapshot(std::shared_ptr<core::InferenceResultPacket> packet,
                              const rules::AlertEvent& event) {
    if (!packet || !packet->source_frame || !packet->source_frame->mat) return;

    try {
        auto now = std::chrono::system_clock::now();
        auto tt = std::chrono::system_clock::to_time_t(now);
        std::tm tm;
        localtime_r(&tt, &tm);
        char buf[64];
        strftime(buf, sizeof(buf), "%Y%m%d_%H%M%S", &tm);

        std::string path = fmt::format("{}/{}_{}_{}.jpg",
            snapshot_dir_, event.rule_type, buf, event.stream_id);
        cv::imwrite(path, *packet->source_frame->mat);
    } catch (...) {
        LOG_ERROR("[AlertNode] Failed to save snapshot");
    }
}

nlohmann::json AlertNode::getStatistics() const {
    nlohmann::json stats;
    for (const auto& r : rules_) {
        stats[r->getName()] = r->getStatistics();
    }
    return stats;
}

REGISTER_NODE("alert", AlertNode)

} // namespace nodes
} // namespace ai_stream