// src/nodes/postprocess/detection_post.cpp
#include "detection_post.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <algorithm>
#include <map>

namespace ai_stream {
namespace nodes {

namespace {

static hal::BBox toHalBox(const core::InferenceResultPacket::BBox& box) {
    hal::BBox out;
    out.x = box.x;
    out.y = box.y;
    out.w = box.w;
    out.h = box.h;
    out.confidence = box.confidence;
    out.class_id = box.class_id;
    out.class_name = box.class_name;
    return out;
}

static core::InferenceResultPacket::BBox toCoreBox(const hal::BBox& box) {
    return core::InferenceResultPacket::BBox(box.x, box.y, box.w, box.h, box.confidence, box.class_id, box.class_name);
}

static void appendRestoredBoxes(
    const std::vector<hal::BBox>& hal_boxes,
    const std::vector<core::InferenceResultPacket::BBox>& original,
    std::vector<core::InferenceResultPacket::BBox>& output) {
    std::vector<bool> used(original.size(), false);
    for (const auto& hal_box : hal_boxes) {
        bool restored = false;
        for (size_t i = 0; i < original.size(); ++i) {
            const auto& original_box = original[i];
            if (used[i] || original_box.x != hal_box.x || original_box.y != hal_box.y ||
                original_box.w != hal_box.w || original_box.h != hal_box.h ||
                original_box.confidence != hal_box.confidence ||
                original_box.class_id != hal_box.class_id) {
                continue;
            }
            used[i] = true;
            output.push_back(original_box);
            restored = true;
            break;
        }
        if (!restored) {
            output.push_back(toCoreBox(hal_box));
        }
    }
}

static bool compareByConfidence(const core::InferenceResultPacket::BBox& a,
                                const core::InferenceResultPacket::BBox& b) {
    return a.confidence > b.confidence;
}

static std::vector<core::InferenceResultPacket::BBox> fallbackClassWiseNms(
    const std::vector<core::InferenceResultPacket::BBox>& input,
    float nms_thresh) {
    std::map<int, std::vector<core::InferenceResultPacket::BBox>> grouped;
    for (const auto& box : input) {
        grouped[box.class_id].push_back(box);
    }

    std::vector<core::InferenceResultPacket::BBox> output;
    for (auto& [class_id, boxes] : grouped) {
        (void)class_id;
        std::sort(boxes.begin(), boxes.end(), compareByConfidence);
        std::vector<bool> suppressed(boxes.size(), false);
        for (size_t i = 0; i < boxes.size(); ++i) {
            if (suppressed[i]) {
                continue;
            }
            output.push_back(boxes[i]);
            for (size_t j = i + 1; j < boxes.size(); ++j) {
                if (suppressed[j]) {
                    continue;
                }
                float x1 = std::max(boxes[i].x, boxes[j].x);
                float y1 = std::max(boxes[i].y, boxes[j].y);
                float x2 = std::min(boxes[i].x + boxes[i].w, boxes[j].x + boxes[j].w);
                float y2 = std::min(boxes[i].y + boxes[i].h, boxes[j].y + boxes[j].h);

                float inter_w = std::max(0.0f, x2 - x1);
                float inter_h = std::max(0.0f, y2 - y1);
                float inter_area = inter_w * inter_h;
                float area_i = boxes[i].w * boxes[i].h;
                float area_j = boxes[j].w * boxes[j].h;
                float union_area = area_i + area_j - inter_area;
                if (union_area > 0 && inter_area / union_area > nms_thresh) {
                    suppressed[j] = true;
                }
            }
        }
    }

    std::sort(output.begin(), output.end(), compareByConfidence);
    return output;
}

} // namespace

DetectionPostProcessNode::DetectionPostProcessNode()
    : core::QueuedNode<IPostprocessNode>("DetectionPostProcess"),
      conf_thresh_(0.5f), nms_thresh_(0.4f), max_detections_(100),
      track_id_enabled_(false), postprocess_type_("detection") {
    LOG_INFO_FMT("[DetectionPostProcessNode] created");
}

DetectionPostProcessNode::~DetectionPostProcessNode() {
    LOG_INFO_FMT("[DetectionPostProcessNode] deinitialized");
}

bool DetectionPostProcessNode::onStartup() {
    accelerator_ = hal::ImageAcceleratorFactory::instance().create(backend_type_);
    if (!accelerator_) {
        LOG_ERROR_FMT("[DetectionPostProcess] Failed to create image accelerator backend");
        return false;
    }

    LOG_INFO_FMT("[DetectionPostProcess] backend: {}", accelerator_->getName());
    return true;
}

void DetectionPostProcessNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (packet->type == core::PacketType::STREAM_END) {
        LOG_INFO_FMT("[DetectionPostProcess] Received stream end");
        broadcast(packet);
        return;
    }

    if (packet->type != core::PacketType::META_DATA) {
        broadcast(packet);
        return;
    }

    auto infer_result = std::static_pointer_cast<core::InferenceResultPacket>(packet);
    auto& dets = infer_result->detections;

    std::vector<core::InferenceResultPacket::BBox> filtered;
    filtered.reserve(dets.size());
    for (const auto& box : dets) {
        if (box.confidence < conf_thresh_) {
            continue;
        }
        if (!class_whitelist_.empty() &&
            std::find(class_whitelist_.begin(), class_whitelist_.end(), box.class_name) == class_whitelist_.end()) {
            continue;
        }
        filtered.push_back(box);
    }

    std::vector<core::InferenceResultPacket::BBox> nms_dets;
    if (accelerator_) {
        std::map<int, std::vector<core::InferenceResultPacket::BBox>> grouped;
        for (const auto& box : filtered) {
            grouped[box.class_id].push_back(box);
        }

        for (auto& [class_id, boxes] : grouped) {
            (void)class_id;
            std::vector<hal::BBox> hal_boxes;
            hal_boxes.reserve(boxes.size());
            for (const auto& box : boxes) {
                hal_boxes.push_back(toHalBox(box));
            }

            if (!accelerator_->nms(hal_boxes, nms_thresh_)) {
                LOG_WARN_FMT("[DetectionPostProcess] HAL NMS failed, using fallback");
                auto fallback = fallbackClassWiseNms(boxes, nms_thresh_);
                nms_dets.insert(nms_dets.end(), fallback.begin(), fallback.end());
                continue;
            }

            appendRestoredBoxes(hal_boxes, boxes, nms_dets);
        }
    } else {
        nms_dets = fallbackClassWiseNms(filtered, nms_thresh_);
    }

    std::sort(nms_dets.begin(), nms_dets.end(), compareByConfidence);

    if (max_detections_ > 0 && static_cast<int>(nms_dets.size()) > max_detections_) {
        nms_dets.resize(max_detections_);
    }

    if (track_id_enabled_) {
        LOG_INFO_FMT("[DetectionPostProcessNode] Track ID enabled - need tracker integration");
    }

    dets = std::move(nms_dets);
    broadcast(infer_result);
}

REGISTER_NODE("detection_post", DetectionPostProcessNode)

} // namespace nodes
} // namespace ai_stream
