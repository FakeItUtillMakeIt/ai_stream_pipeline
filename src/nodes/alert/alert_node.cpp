// src/nodes/alert/alert_node.cpp
#include "alert_node.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "rules/alert/alert_rule_factory.h"
#include "rules/alert/absence_rule.h"
#include "rules/alert/climbing_rule.h"
#include "rules/alert/fall_down_rule.h"
#include "rules/alert/human_gathering_rule.h"
#include "rules/alert/missing_helmet_rule.h"
#include "rules/alert/missing_safety_belt_rule.h"
#include "rules/alert/person_intrusion_rule.h"
#include "rules/alert/phone_call_rule.h"
#include "rules/alert/sleeping_on_duty_rule.h"
#include "rules/alert/smoking_rule.h"
#include "rules/alert/fire_lane_occupancy_rule.h"
#include "rules/alert/discover_crystal_rule.h"
#include "rules/alert/discover_visible_fire_rule.h"
#include "rules/alert/discover_smoke_rule.h"
#include "rules/alert/discover_hose_cutoff_rule.h"

#include "3rd_party/log_mgr/log_mgr.h"
#include <opencv2/opencv.hpp>
#include <future>

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
    std::lock_guard<std::mutex> lock(mutex_);
    running_ = true;
    if (!snapshot_dir_.empty()) {
        std::filesystem::create_directories(snapshot_dir_);
    }
    LOG_INFO_FMT("[AlertNode] Started with {} rules", rules_.size());
    return true;
}

void AlertNode::stop() {
    running_ = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& r : rules_) r->reset();
    }
    LOG_INFO("[AlertNode] Stopped");
}

void AlertNode::pushData(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END)
    {
        LOG_INFO_FMT("[Alert] Received stream end");
        stop();
        broadcast(packet);
        return;
    }
    

    if (!running_) return;
    if (packet->type != core::PacketType::META_DATA) return;

    auto infer_packet = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
    auto all_alert_results =process_all_alerts_sequence(infer_packet);
    for (auto& r : all_alert_results) {
        if (r.alert_events.empty()) continue;
        for (auto& e : r.alert_events) {
            LOG_INFO_FMT("[AlertNode] Alert: {}-{}-status({}-{})", e.alert_name, rules::alertTypeMap[e.alert_type],static_cast<int>(e.status),e.description);
        }
    }
    infer_packet->alert_result = std::move(all_alert_results);
    broadcast(infer_packet);
}

void AlertNode::addRule(rules::AlertRulePtr rule) {
    std::lock_guard<std::mutex> lock(mutex_);
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

std::vector<rules::AlertResult> AlertNode::process_all_alerts_parallel(std::shared_ptr<core::InferenceResultPacket> packet)
{
    std::vector<rules::AlertResult> all_alert_results;
    
    std::vector<rules::AlertRulePtr> rules_snapshot;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        rules_snapshot = rules_;
    }
    
    all_alert_results.reserve(rules_snapshot.size());
    std::vector<std::future<rules::AlertResult>> futures;
    futures.reserve(rules_snapshot.size());
    
    size_t task_count =0;
    for (auto& rule : rules_snapshot)
    {
        auto rule_copy = rule;
        auto future = std::async(std::launch::async, [this, rule_copy, packet]() {
            return process_single_alert(rule_copy, packet);
        });
        futures.push_back(std::move(future));
        task_count++;
    }
    
    LOG_INFO_FMT("[AlertNode] Submitted {} tasks to future", task_count);
    
    size_t task_done = 0;
    for (auto& future : futures) { 
        try
        {
            all_alert_results.push_back(future.get());
            task_done++;
        }
        catch(const std::exception& e)
        {
            LOG_ERROR_FMT("[AlertNode] Failed to process alert task: {}", e.what());
            rules::AlertResult failed_result;
            failed_result.rule_name = "Unknown";
            failed_result.rule_status = rules::RuleStatus::RULE_STATUS_FAIL;
            failed_result.alert_type = rules::AlertType::ALERT_UNKNOWN;
            failed_result.error_message= e.what();
            all_alert_results.push_back(failed_result);
            task_done++;
        }
    }
    LOG_INFO_FMT("[AlertNode] Completed {} tasks", task_done);
    return all_alert_results;
}

std::vector<rules::AlertResult> AlertNode::process_all_alerts_sequence(std::shared_ptr<core::InferenceResultPacket> packet)
{
    auto infer = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
    if (!infer) return {};

    std::vector<rules::AlertResult> all_alert_results;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& rule : rules_) {
            rules::AlertResult alert_result = process_single_alert(rule, packet);
            // if (!alert_result.alert_events.empty()) {
            //     for (const auto& e : alert_result.alert_events) {
            //         if (e.status == rules::AlertStatus::ALERT_STATUS_OCCUR) {
            //             saveSnapshot(infer, e);
            //         }
            //     }
            // }
            all_alert_results.push_back(alert_result);
        }
    }
    return all_alert_results;
}

rules::AlertResult AlertNode::process_single_alert(rules::AlertRulePtr rule,
                                    std::shared_ptr<core::InferenceResultPacket> packet)
{
    rules::AlertResult alert_result;
    alert_result.rule_name = rule->getName();
    auto start_time = utils::TimeUtil::currentTimeMs();
    try
    {
        alert_result.rule_status = rule->process(packet, alert_result);
        alert_result.process_time_ms = utils::TimeUtil::currentTimeMs() - start_time;
        if(alert_result.rule_status == rules::RuleStatus::RULE_STATUS_OK)
            LOG_INFO_FMT("Alert type:{} - {} processed success: {} ms", rules::alertTypeMap[rule->getType()] ,rule->getName(), alert_result.process_time_ms);
        else
            LOG_WARN_FMT("Alert {} processed failed, status code: {}", rule->getName(), static_cast<int>(alert_result.rule_status));
    }
    catch(const std::exception& e)
    {
        alert_result.rule_status = rules::RuleStatus::RULE_STATUS_FAIL;
        LOG_ERROR_FMT("Failed to process alert {}: {}", rule->getName() ,e.what());
    }
    catch(...)
    {
        alert_result.rule_status = rules::RuleStatus::RULE_STATUS_FAIL;
        LOG_ERROR_FMT("Failed to process alert {}", rule->getName());
    }

    return alert_result;
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
        LOG_INFO_FMT("[AlertNode] {}: {}", e.alert_name, e.description);

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

        std::string path = fmt::format("{}/{}_{}.jpg",
            snapshot_dir_, event.alert_name, buf);
        cv::imwrite(path, *packet->source_frame->mat);
    } catch (...) {
        LOG_ERROR("[AlertNode] Failed to save snapshot");
    }
}

nlohmann::json AlertNode::getStatistics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    nlohmann::json stats;
    for (const auto& r : rules_) {
        stats[r->getName()] = r->getStatistics();
    }
    return stats;
}

REGISTER_NODE("alert", AlertNode)

} // namespace nodes
} // namespace ai_stream