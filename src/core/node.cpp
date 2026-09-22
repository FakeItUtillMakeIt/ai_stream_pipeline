// src/core/node.cpp
#include "ai_stream/core/node.h"
#include "ai_stream/core/pipeline.h" // 完整定义
#include <algorithm>

namespace ai_stream {
namespace core {

namespace {

static std::shared_ptr<BasePacket> trimGpuPayload(const std::shared_ptr<BasePacket>& packet);

static bool videoFrameHasGpuPayload(const VideoFramePacket& frame) {
    return frame.is_gpu || frame.d_ptr != nullptr || frame.d_bgr_ptr != nullptr ||
           frame.d_data_uv != nullptr;
}

static std::shared_ptr<VideoFramePacket> trimVideoFrameGpu(const std::shared_ptr<VideoFramePacket>& frame) {
    if (!frame) return nullptr;
    if (!videoFrameHasGpuPayload(*frame)) return frame;
    auto trimmed = std::make_shared<VideoFramePacket>(*frame);
    trimmed->is_gpu = false;
    trimmed->d_ptr = nullptr;
    trimmed->d_width = 0;
    trimmed->d_height = 0;
    trimmed->d_pitch = 0;
    trimmed->d_buf_owner.reset();
    trimmed->d_data_uv = nullptr;
    trimmed->d_pitch_uv = 0;
    trimmed->d_uv_owner.reset();
    trimmed->d_bgr_ptr = nullptr;
    trimmed->d_bgr_pitch = 0;
    trimmed->d_bgr_height = 0;
    trimmed->d_bgr_width = 0;
    trimmed->d_bgr_owner.reset();
    return trimmed;
}

static std::shared_ptr<InferenceResultPacket> trimInferenceGpu(const std::shared_ptr<InferenceResultPacket>& packet) {
    if (!packet) return nullptr;
    if (!packet->source_frame || !videoFrameHasGpuPayload(*packet->source_frame)) return packet;
    auto trimmed = std::make_shared<InferenceResultPacket>(*packet);
    if (trimmed->source_frame) {
        auto source_trimmed = std::dynamic_pointer_cast<VideoFramePacket>(trimmed->source_frame);
        if (source_trimmed) {
            trimmed->source_frame = trimVideoFrameGpu(source_trimmed);
        }
    }
    return trimmed;
}

static std::shared_ptr<BasePacket> trimGpuPayload(const std::shared_ptr<BasePacket>& packet) {
    if (!packet) return nullptr;
    if (auto frame = std::dynamic_pointer_cast<VideoFramePacket>(packet)) {
        return trimVideoFrameGpu(frame);
    }
    if (auto infer = std::dynamic_pointer_cast<InferenceResultPacket>(packet)) {
        return trimInferenceGpu(infer);
    }
    return packet;
}

} // namespace

void Node::broadcast(std::shared_ptr<BasePacket> packet) {
    if (!packet) return;
    auto it = packet->cost_time_map.find(name_);
    if (it != packet->cost_time_map.end()) {
        recordMetricsImpl(it->second, false);
    }
    // 打戳生产者：下游可据此区分数据来源（如 fusion 区分多路推理结果）
    packet->producer_id = name_;

    // 统计有效下游数：>1 时每个分支使用独立副本，避免各下游 worker 并发修改同一包。
    size_t valid_count = 0;
    for (const auto& weak_down : downstreams_) {
        if (!weak_down.node.expired()) ++valid_count;
    }

    bool has_expired = false;
    size_t valid_idx = 0;
    for (auto& weak_down : downstreams_) {
        auto down = weak_down.node.lock();
        if (!down) {
            has_expired = true;
            continue;
        }
        // 最后一个有效分支复用原包，其余分支克隆（图像负载共享、容器深拷贝）
        std::shared_ptr<BasePacket> branch = packet;
        if (valid_count > 1 && valid_idx < valid_count - 1) {
            branch = packet->cloneForBranch();
        }
        ++valid_idx;

        if (weak_down.gpu_needed) {
            down->pushData(branch);
        } else {
            down->pushData(trimGpuPayload(branch));
        }
    }
    // 清理已失效的下游弱引用（构建完成后 downstreams_ 无并发写入）
    if (has_expired) {
        downstreams_.erase(
            std::remove_if(downstreams_.begin(), downstreams_.end(),
                           [](const DownstreamRef& w) { return w.node.expired(); }),
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
