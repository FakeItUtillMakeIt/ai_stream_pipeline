// src/nodes/fusion/fusion_node.cpp
#include "fusion_node.h"
#include "utils/time_util.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include "registry/node_factory.h"
#include <algorithm>
#include <cstdlib>

namespace ai_stream
{
    namespace nodes
    {

        namespace
        {
            float boxIoU(const core::InferenceResultPacket::BBox& a,
                         const core::InferenceResultPacket::BBox& b)
            {
                float ix1 = std::max(a.x, b.x);
                float iy1 = std::max(a.y, b.y);
                float ix2 = std::min(a.x + a.w, b.x + b.w);
                float iy2 = std::min(a.y + a.h, b.y + b.h);
                if (ix2 <= ix1 || iy2 <= iy1) return 0.0f;
                float inter = (ix2 - ix1) * (iy2 - iy1);
                float uni = a.w * a.h + b.w * b.h - inter;
                return uni > 0.0f ? inter / uni : 0.0f;
            }

            // 同名类别框按置信度贪心 NMS（跨推理源去重）
            void applyCrossNms(std::vector<core::InferenceResultPacket::BBox>& dets, float iou_threshold)
            {
                std::sort(dets.begin(), dets.end(),
                          [](const core::InferenceResultPacket::BBox& a, const core::InferenceResultPacket::BBox& b)
                          { return a.confidence > b.confidence; });
                std::vector<bool> suppressed(dets.size(), false);
                for (size_t i = 0; i < dets.size(); ++i)
                {
                    if (suppressed[i]) continue;
                    for (size_t j = i + 1; j < dets.size(); ++j)
                    {
                        if (suppressed[j]) continue;
                        if (dets[i].class_name != dets[j].class_name) continue;
                        if (boxIoU(dets[i], dets[j]) > iou_threshold)
                        {
                            suppressed[j] = true;
                        }
                    }
                }
                std::vector<core::InferenceResultPacket::BBox> kept;
                kept.reserve(dets.size());
                for (size_t i = 0; i < dets.size(); ++i)
                {
                    if (!suppressed[i]) kept.push_back(dets[i]);
                }
                dets = std::move(kept);
            }
        } // namespace

        FusionNodeImpl::FusionNodeImpl() : core::QueuedNode<IFusionNode>("Fusion") {}

        bool FusionNodeImpl::configureImpl(const std::string& node_id, const nlohmann::json& params)
        {
            (void)node_id;

            std::string mode = params.value("mode", "action");
            if (mode == "detection_merge")
            {
                fusion_mode_ = FusionMode::DETECTION_MERGE;
            }
            else if (mode == "object_level")
            {
                fusion_mode_ = FusionMode::OBJECT_LEVEL;
            }
            else
            {
                fusion_mode_ = FusionMode::FRAME_LEVEL;
            }

            if (params.contains("action_source"))
            {
                action_source_ = params["action_source"].get<std::string>();
            }
            if (params.contains("timestamp_threshold_ms"))
            {
                timestamp_threshold_ms_ = params["timestamp_threshold_ms"].get<int64_t>();
            }

            if (fusion_mode_ == FusionMode::DETECTION_MERGE)
            {
                if (!params.contains("detection_sources") || !params["detection_sources"].is_array() ||
                    params["detection_sources"].empty())
                {
                    LOG_ERROR_FMT("[Fusion] detection_merge mode requires non-empty 'detection_sources'");
                    return false;
                }
                merge_sources_ = params["detection_sources"].get<std::vector<std::string>>();

                if (params.contains("class_offsets") && params["class_offsets"].is_object())
                {
                    for (auto it = params["class_offsets"].begin(); it != params["class_offsets"].end(); ++it)
                    {
                        class_offsets_[it.key()] = it.value().get<int>();
                    }
                }
                wait_timeout_ms_ = params.value("wait_timeout_ms", wait_timeout_ms_);
                cross_nms_ = params.value("cross_nms", cross_nms_);
                nms_iou_threshold_ = params.value("nms_iou_threshold", nms_iou_threshold_);

                std::string sources_str;
                for (const auto& s : merge_sources_) sources_str += s + " ";
                LOG_INFO_FMT("[Fusion] detection_merge: sources=[{}], offsets={}, wait_timeout={}ms, cross_nms={}",
                             sources_str, class_offsets_.size(), wait_timeout_ms_, cross_nms_);
            }
            return true;
        }

        bool FusionNodeImpl::onStartup()
        {
            LOG_INFO_FMT("[Fusion] Started with mode: {}",
                         fusion_mode_ == FusionMode::FRAME_LEVEL ? "FRAME_LEVEL"
                         : fusion_mode_ == FusionMode::OBJECT_LEVEL ? "OBJECT_LEVEL"
                         : "DETECTION_MERGE");
            return true;
        }

        void FusionNodeImpl::onIdle()
        {
            // 无新包到达时也按超时刷出待合并帧（部分合并）
            std::lock_guard<std::mutex> lock(mutex_);
            if (!pending_.empty())
            {
                flushExpiredPending(utils::TimeUtil::currentTimeMs());
            }
        }

        void FusionNodeImpl::onShutdown()
        {
            LOG_INFO_FMT("[Fusion] Stopped");
        }

        void FusionNodeImpl::processPacket(std::shared_ptr<core::BasePacket> packet)
        {
            // STREAM_END：排空所有待合并帧后自停并转发
            if (packet->type == core::PacketType::STREAM_END)
            {
                LOG_INFO_FMT("[Fusion] Received stream end");
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    while (!pending_.empty())
                    {
                        mergeAndBroadcast(pending_.begin()->first);
                    }
                    has_cached_action_ = false;
                }
                // 不在此处调用 stop()，避免从 worker 线程调用导致自连接死锁
                // running_ 会在 workerLoop 中检查，worker 线程会自然退出
                broadcast(packet);
                return;
            }

            auto result = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
            if (!result)
            {
                // 非推理结果包，直接转发
                broadcast(packet);
                return;
            }

            std::lock_guard<std::mutex> lock(mutex_);

            // 动作识别结果：任何模式下都更新缓存（兼容原动作融合）
            if (!result->action_results.empty())
            {
                handleActionResult(result);
                return;
            }

            // DETECTION_MERGE：配置来源的检测结果进入帧配对
            if (fusion_mode_ == FusionMode::DETECTION_MERGE && isMergeSource(result->producer_id))
            {
                handleDetectionMerge(result);
                return;
            }

            if (fusion_mode_ == FusionMode::DETECTION_MERGE)
            {
                // 未配置来源的结果包原样透传
                broadcast(result);
                return;
            }

            // 原有动作融合逻辑：检测（通常来自tracker）附加缓存的动作结果
            if (!result->detections.empty())
            {
                handleDetectionResult(result);
            }
        }

        // IFusionNode接口实现
        void FusionNodeImpl::setFusionMode(FusionMode mode)
        {
            fusion_mode_ = mode;
        }

        FusionMode FusionNodeImpl::getFusionMode() const
        {
            return fusion_mode_;
        }

        void FusionNodeImpl::setActionSource(const std::string &source_node_id)
        {
            action_source_ = source_node_id;
        }

        std::string FusionNodeImpl::getActionSource() const
        {
            return action_source_;
        }

        void FusionNodeImpl::setTimestampThreshold(int64_t threshold_ms)
        {
            timestamp_threshold_ms_ = threshold_ms;
        }

        int64_t FusionNodeImpl::getTimestampThreshold() const
        {
            return timestamp_threshold_ms_;
        }

        void FusionNodeImpl::handleActionResult(std::shared_ptr<core::InferenceResultPacket> result)
        {
            // 缓存最新的动作识别结果
            cached_action_ = result->action_results[0];
            cached_action_timestamp_ = result->timestamp_ms;
            has_cached_action_ = true;

            LOG_DEBUG_FMT("[Fusion] Cached action: {} (confidence: {:.4f}, timestamp: {}ms)",
                          cached_action_.action_label, cached_action_.confidence, cached_action_timestamp_);
        }

        void FusionNodeImpl::handleDetectionResult(std::shared_ptr<core::InferenceResultPacket> result)
        {
            // 检查是否有缓存的动作结果
            if (!has_cached_action_)
            {
                // 没有动作识别结果，直接转发检测结果
                broadcast(result);
                return;
            }

            // 检查时间戳是否在阈值范围内
            int64_t time_diff = std::abs(result->timestamp_ms - cached_action_timestamp_);
            if (time_diff > timestamp_threshold_ms_)
            {
                // 动作识别结果太旧，不融合
                LOG_DEBUG_FMT("[Fusion] Action result too old (diff: {}ms > threshold: {}ms), skipping",
                              time_diff, timestamp_threshold_ms_);
                broadcast(result);
                return;
            }

            // 融合：将动作结果附加到检测结果
            result->action_results.push_back(cached_action_);

            LOG_DEBUG_FMT("[Fusion] Fused detection with action: {} (time_diff: {}ms)",
                          cached_action_.action_label, time_diff);

            // 广播融合后的结果
            broadcast(result);
        }

        bool FusionNodeImpl::isMergeSource(const std::string& producer) const
        {
            return std::find(merge_sources_.begin(), merge_sources_.end(), producer) != merge_sources_.end();
        }

        void FusionNodeImpl::handleDetectionMerge(std::shared_ptr<core::InferenceResultPacket> result)
        {
            auto key = std::make_pair(result->stream_id, result->frame_id);
            int64_t now_ms = utils::TimeUtil::currentTimeMs();

            auto& pending = pending_[key];
            if (pending.first_arrival_ms == 0)
            {
                pending.first_arrival_ms = now_ms;
            }
            pending.per_source[result->producer_id] = result;

            // 全部来源到齐 → 立即合并
            if (pending.per_source.size() >= merge_sources_.size())
            {
                mergeAndBroadcast(key);
            }

            // 清理超时条目（部分合并广播）
            flushExpiredPending(now_ms);

            // 容量保护：强制合并最旧条目
            while (pending_.size() > MAX_PENDING_FRAMES)
            {
                LOG_WARN_FMT("[Fusion] Pending frames overflow, force merging oldest");
                mergeAndBroadcast(pending_.begin()->first);
            }
        }

        void FusionNodeImpl::flushExpiredPending(int64_t now_ms)
        {
            std::vector<std::pair<uint32_t, int64_t>> expired;
            for (const auto& [key, frame] : pending_)
            {
                if (now_ms - frame.first_arrival_ms > wait_timeout_ms_)
                {
                    expired.push_back(key);
                }
            }
            for (const auto& key : expired)
            {
                mergeAndBroadcast(key);
            }
        }

        void FusionNodeImpl::mergeAndBroadcast(const std::pair<uint32_t, int64_t>& key)
        {
            auto it = pending_.find(key);
            if (it == pending_.end()) return;

            PendingFrame frame = std::move(it->second);
            pending_.erase(it);

            // 按配置顺序收集各来源结果，保证输出稳定
            std::vector<std::shared_ptr<core::InferenceResultPacket>> parts;
            std::string missing;
            for (const auto& source : merge_sources_)
            {
                auto pit = frame.per_source.find(source);
                if (pit != frame.per_source.end())
                {
                    parts.push_back(pit->second);
                }
                else
                {
                    missing += source + " ";
                }
            }
            // 不在配置顺序中的多余来源（理论上没有）也并入
            for (auto& [producer, pkt] : frame.per_source)
            {
                if (!isMergeSource(producer)) continue;
                bool already = false;
                for (const auto& p : parts)
                {
                    if (p == pkt) { already = true; break; }
                }
                if (!already) parts.push_back(pkt);
            }

            if (parts.empty()) return;

            if (!missing.empty())
            {
                LOG_DEBUG_FMT("[Fusion] Partial merge for stream {} frame {}: missing [{}]",
                              key.first, key.second, missing);
            }

            // 以首包为基础构造合并结果
            auto merged = std::make_shared<core::InferenceResultPacket>();
            const auto& first = *parts.front();
            merged->stream_id = first.stream_id;
            merged->source_id = first.source_id;
            merged->frame_id = first.frame_id;
            merged->timestamp_ms = first.timestamp_ms;
            merged->source_frame = first.source_frame;

            size_t total_dets = 0;
            for (const auto& part : parts)
            {
                total_dets += part->detections.size();
            }
            merged->detections.reserve(total_dets);

            for (const auto& part : parts)
            {
                // 可选 class_id 偏移（class_name 保持不变，下游规则按名匹配）
                int offset = 0;
                auto oit = class_offsets_.find(part->producer_id);
                if (oit != class_offsets_.end())
                {
                    offset = oit->second;
                }

                for (auto det : part->detections)
                {
                    det.class_id += offset;
                    merged->detections.push_back(std::move(det));
                }

                merged->pose_results.insert(merged->pose_results.end(),
                                            part->pose_results.begin(), part->pose_results.end());
                merged->action_results.insert(merged->action_results.end(),
                                              part->action_results.begin(), part->action_results.end());

                for (const auto& [node_name, cost] : part->cost_time_map)
                {
                    merged->cost_time_map[node_name] = cost;
                }
                merged->cost_ms = std::max(merged->cost_ms, part->cost_ms);
            }

            // 可选跨源 NMS（同名类别去重，默认关）
            if (cross_nms_)
            {
                applyCrossNms(merged->detections, nms_iou_threshold_);
            }

            // 兼容动作融合：附加时间戳匹配的动作识别缓存
            if (has_cached_action_)
            {
                int64_t time_diff = std::abs(merged->timestamp_ms - cached_action_timestamp_);
                if (time_diff <= timestamp_threshold_ms_)
                {
                    merged->action_results.push_back(cached_action_);
                }
            }

            LOG_DEBUG_FMT("[Fusion] Merged {} sources -> {} detections (stream {}, frame {})",
                          parts.size(), merged->detections.size(), key.first, key.second);
            broadcast(merged);
        }

        // 注册节点
        REGISTER_NODE("fusion", FusionNodeImpl)

    } // namespace nodes
} // namespace ai_stream
