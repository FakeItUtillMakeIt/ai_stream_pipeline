// src/nodes/postprocess/detection_post.cpp
#include "detection_post.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <algorithm>

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

        void DetectionPostProcessNode::pushData(std::shared_ptr<core::BasePacket> packet)
        {
            if (packet->type != core::PacketType::META_DATA)
            {
                broadcast(packet);
                return;
            }

            auto infer_result = std::static_pointer_cast<core::InferenceResultPacket>(packet);

            // 1. 置信度过滤
            auto &dets = infer_result->detections;
            dets.erase(std::remove_if(dets.begin(), dets.end(),
                                      [this](const auto &box)
                                      { return box.confidence < conf_thresh_; }),
                       dets.end());

            // 2. NMS（简化，实际需实现 IoU 计算）
            // ...

            // 广播后处理后的结果
            broadcast(infer_result);
        }

        REGISTER_NODE("detection_post", DetectionPostProcessNode)

    } // namespace nodes
} // namespace ai_stream