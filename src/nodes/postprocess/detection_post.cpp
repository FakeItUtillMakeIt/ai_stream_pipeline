// src/nodes/postprocess/detection_post.cpp
#include "detection_post.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <algorithm>
#include <cmath>
#include <map>

namespace ai_stream
{
    namespace nodes
    {

        DetectionPostProcessNode::DetectionPostProcessNode() : IPostprocessNode("DetectionPostProcess"),
                                                               conf_thresh_(0.5f), nms_thresh_(0.4f), max_detections_(100),
                                                               track_id_enabled_(false), postprocess_type_("detection")
        {
            LOG_INFO_FMT("[DetectionPostProcessNode] created");
        }

        DetectionPostProcessNode::~DetectionPostProcessNode()
        {
            LOG_INFO_FMT("[DetectionPostProcessNode] deinitialized");
        }

        namespace
        {
            // IoU 计算辅助函数
            float calculateIoU(const core::InferenceResultPacket::BBox &box1, const core::InferenceResultPacket::BBox &box2)
            {
                float x1 = std::max(box1.x, box2.x);
                float y1 = std::max(box1.y, box2.y);
                float x2 = std::min(box1.x + box1.w, box2.x + box2.w);
                float y2 = std::min(box1.y + box1.h, box2.y + box2.h);

                float inter_area = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
                float box1_area = box1.w * box1.h;
                float box2_area = box2.w * box2.h;

                if (box1_area + box2_area - inter_area <= 0.0f)
                {
                    return 0.0f;
                }
                return inter_area / (box1_area + box2_area - inter_area);
            }

            // 按置信度排序
            bool compareByConfidence(const core::InferenceResultPacket::BBox &a, const core::InferenceResultPacket::BBox &b)
            {
                return a.confidence > b.confidence;
            }
        } // namespace

        void DetectionPostProcessNode::pushData(std::shared_ptr<core::BasePacket> packet)
        {
            if (packet->type == core::PacketType::STREAM_END)
            {
                LOG_INFO_FMT("[DetectionPostProcess] Received stream end");
                stop();
                broadcast(packet);
                return;
            }
            if (packet->type != core::PacketType::META_DATA)
            {
                broadcast(packet);
                return;
            }

            auto infer_result = std::static_pointer_cast<core::InferenceResultPacket>(packet);
            auto &dets = infer_result->detections;

            // 1. 置信度过滤
            dets.erase(std::remove_if(dets.begin(), dets.end(),
                                      [this](const auto &box)
                                      {
                                          if (box.confidence < conf_thresh_)
                                          {
                                              return true;
                                          }
                                          // 类别白名单过滤
                                          if (!class_whitelist_.empty() &&
                                              std::find(class_whitelist_.begin(), class_whitelist_.end(), box.class_name) == class_whitelist_.end())
                                          {
                                              return true;
                                          }
                                          return false;
                                      }),
                       dets.end());

            // 2. NMS（非极大值抑制）
            std::sort(dets.begin(), dets.end(), compareByConfidence);
            std::vector<core::InferenceResultPacket::BBox> nms_dets;
            std::vector<bool> suppressed(dets.size(), false);

            for (size_t i = 0; i < dets.size(); ++i)
            {
                if (suppressed[i])
                    continue;

                nms_dets.push_back(dets[i]);

                for (size_t j = i + 1; j < dets.size(); ++j)
                {
                    if (suppressed[j])
                        continue;

                    // 同一类别的框才进行 NMS
                    if (dets[i].class_id == dets[j].class_id)
                    {
                        float iou = calculateIoU(dets[i], dets[j]);
                        if (iou > nms_thresh_)
                        {
                            suppressed[j] = true;
                        }
                    }
                }
            }

            dets = std::move(nms_dets);

            // 3. 最大检测数限制
            if (max_detections_ > 0 && static_cast<int>(dets.size()) > max_detections_)
            {
                dets.resize(max_detections_);
            }

            // 4. 跟踪ID处理（占位，实际需集成跟踪器）
            if (track_id_enabled_)
            {
                LOG_INFO_FMT("[DetectionPostProcessNode] Track ID enabled - need tracker integration");
            }

            // 广播后处理后的结果
            broadcast(infer_result);
        }

        REGISTER_NODE("detection_post", DetectionPostProcessNode)

    } // namespace nodes
} // namespace ai_stream