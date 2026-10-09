// src/nodes/gate/report_sink_node.h
#pragma once

#include "ai_stream/core/queued_node.h"
#include "ai_stream/nodes/i_report_node.h"
#include "ai_stream/core/packet.h"

#include <string>

namespace ai_stream {
namespace nodes {

/**
 * @brief 告警上报出口：收到即逐个 POST 到 webhook。
 *        上游是 evidence(无 gate) 还是 vlm_gate(有 gate) 由管道接线决定，
 *        本节点无条件逻辑——只上报它收到的事件。
 */
class ReportSinkNode : public core::QueuedNode<IReportNode> {
public:
    ReportSinkNode();
    ~ReportSinkNode() override;

    void setReportConfig(const ReportConfig& config) override;

protected:
    bool onStartup() override;
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;

private:
    ReportConfig cfg_;
};

}  // namespace nodes
}  // namespace ai_stream
