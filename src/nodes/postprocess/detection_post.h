// src/nodes/postprocess/detection_post.h
#pragma once

#include "ai_stream/nodes/i_postprocess_node.h"
#include "ai_stream/core/queued_node.h"

namespace ai_stream
{
    namespace nodes
    {

        class DetectionPostProcessNode : public core::QueuedNode<IPostprocessNode>
        {
        public:
            DetectionPostProcessNode();
            ~DetectionPostProcessNode();

            void processPacket(std::shared_ptr<core::BasePacket> packet) override;

            // 配置参数
            void setConfidenceThreshold(float threshold) { conf_thresh_ = threshold; }
            float getConfidenceThreshold() const override { return conf_thresh_; }
            void setNmsThreshold(float threshold) { nms_thresh_ = threshold; }
            float getNmsThreshold() const override { return nms_thresh_; }
            void setMaxDetections(int max_detections) { max_detections_ = max_detections; }

            void setClassWhitelist(const std::vector<std::string> &class_names) override { class_whitelist_ = class_names; }
            void setTrackIdEnabled(bool enable) override { track_id_enabled_ = enable; }
            void setPostProcessType(const std::string &type) override { postprocess_type_ = type; }
            std::string getPostProcessType() const override { return postprocess_type_; }

        private:
            float conf_thresh_ = 0.5f;
            float nms_thresh_ = 0.4f;
            int max_detections_ = 100;
            std::vector<std::string> class_whitelist_;
            bool track_id_enabled_ = false;
            std::string postprocess_type_ = "detection";
        };

    } // namespace nodes
} // namespace ai_stream