// src/nodes/gate/vlm_gate_node.h
#pragma once

#include "ai_stream/core/queued_node.h"
#include "ai_stream/nodes/i_vlm_gate_node.h"
#include "ai_stream/core/packet.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>

namespace ai_stream {
namespace nodes {

/**
 * @brief 告警闸门：收到 evidence 转发的"已标注快照"告警后，调 VLM 判真伪，
 *        仅把判真的事件放行给下游 report。判假/存疑/异常一律不下发。
 */
class VlmGateNode : public core::QueuedNode<IVlmGateNode> {
public:
    VlmGateNode();
    ~VlmGateNode() override;

    void setGateConfig(const VlmGateConfig& config) override;

protected:
    bool onStartup() override;
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;

private:
    // 单次核验结果
    struct Verdict {
        bool pushed = false;       // 是否放行下游
        std::string decision;      // PUSH / DROP / HOLD / PASSTHROUGH
        bool verdict = false;      // VLM 判真
        float confidence = 0.0f;
        std::string reason;
        std::string error;         // 非空表示没判成（超时/非法JSON/无图等）→ HOLD
        long long latency_ms = 0;
    };

    // 读 snapshot 文件 → base64（超 max_side 先下采样）。返回空表示图不可用。
    std::string loadSnapshotBase64(const std::string& path, int max_side);

    // 三种处理模式
    enum class GateMode { Verify, Passthrough, Hold };

    // 按 alert_name 解析出"有效提示词"（question/system/阈值/max_tokens 全部落实）。
    // 命中 cfg_.prompts 用精调；否则按 unmatched_policy：Auto=用模板套 alert_name 生成，
    // Passthrough/Hold 直接返回对应模式（out 不用于调 VLM）。
    GateMode resolvePrompt(const std::string& alert_name, VlmPromptSpec& out);

    // 调 OpenAI 兼容端点，解析出 verdict/confidence/reason。失败填 error。
    Verdict callVlm(const std::string& image_b64, const VlmPromptSpec& spec);

    // 冷却判定 + 记录。命中冷却返回 true（本周期已决策，跳过重复调用）。
    bool inCooldown(const std::string& key, long long now_ms);
    void touchCooldown(const std::string& key, long long now_ms);

    void appendAudit(const core::InferenceResultPacket& pkt, const rules::AlertEvent& ev,
                     const std::string& snap_path, const Verdict& v);

    VlmGateConfig cfg_;
    std::string review_path_;

    std::mutex cooldown_mutex_;
    std::unordered_map<std::string, long long> last_decision_ms_;
};

}  // namespace nodes
}  // namespace ai_stream
