// src/nodes/infer/relation_recognition.cpp
#include "relation_recognition.h"

#include "3rd_party/log_mgr/log_mgr.h"
#include "ai_stream/core/packet.h"
#include "registry/node_factory.h"
#include "utils/time_util.h"

#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>

namespace ai_stream {
namespace nodes {

namespace {
constexpr int kLetterboxPad = 114;   // 与检测侧/关系模型训练一致的灰底
}

RelationRecognitionNode::RelationRecognitionNode() = default;

RelationRecognitionNode::~RelationRecognitionNode() {
    if (engine_) {
        engine_->unload();
        engine_.reset();
    }
}

void RelationRecognitionNode::setModelPath(const std::string& path) { cfg_.model_path = path; }
void RelationRecognitionNode::setPredicateBank(const std::string& path) { cfg_.predicate_bank_path = path; }
void RelationRecognitionNode::setImageSize(int size) { cfg_.image_size = size; }
void RelationRecognitionNode::setPredicates(const std::vector<std::string>& p) { cfg_.predicates = p; }
void RelationRecognitionNode::setScoreThreshold(float t) {
    cfg_.score_threshold = t;
    if (engine_)
        engine_->setScoreThreshold(t);
}
void RelationRecognitionNode::setBoundaryClasses(const std::vector<std::string>& c) { cfg_.boundary_classes = c; }
void RelationRecognitionNode::setBoxConfidence(float c) { cfg_.box_confidence = c; }
void RelationRecognitionNode::setMaxBoxes(int n) { cfg_.max_boxes = n; }

bool RelationRecognitionNode::configureImpl(const std::string& node_id,
                                           const nlohmann::json& params) {
    node_id_ = node_id;
    try {
        if (params.contains("model_path"))
            cfg_.model_path = params["model_path"].get<std::string>();
        if (params.contains("predicate_bank"))
            cfg_.predicate_bank_path = params["predicate_bank"].get<std::string>();
        if (params.contains("predicate_bank_path"))
            cfg_.predicate_bank_path = params["predicate_bank_path"].get<std::string>();
        if (params.contains("img_size"))
            cfg_.image_size = params["img_size"].get<int>();
        if (params.contains("max_boxes"))
            cfg_.max_boxes = params["max_boxes"].get<int>();
        if (params.contains("predicates"))
            cfg_.predicates = params["predicates"].get<std::vector<std::string>>();
        if (params.contains("per_predicate_threshold")) {
            cfg_.per_predicate_threshold.clear();
            for (const auto& kv : params["per_predicate_threshold"].items())
                cfg_.per_predicate_threshold.emplace_back(
                    kv.key(), kv.value().get<float>());
        }
        if (params.contains("boundary_classes"))
            cfg_.boundary_classes = params["boundary_classes"].get<std::vector<std::string>>();
        if (params.contains("score_threshold"))
            cfg_.score_threshold = params["score_threshold"].get<float>();
        if (params.contains("box_confidence"))
            cfg_.box_confidence = params["box_confidence"].get<float>();
        if (params.contains("max_relations"))
            cfg_.max_relations = params["max_relations"].get<int>();
        if (params.contains("run_every_n_frames"))
            cfg_.run_every_n_frames = std::max(1, params["run_every_n_frames"].get<int>());
        if (params.contains("device_id"))
            cfg_.device_id = params["device_id"].get<int>();
        if (params.contains("backend")) {
            const std::string b = params["backend"].get<std::string>();
            if (b == "tensorrt") backend_ = hal::RelationBackend::TENSORRT;
            else if (b == "onnxruntime") backend_ = hal::RelationBackend::ONNXRUNTIME;
            else if (b == "cpu") backend_ = hal::RelationBackend::CPU;
            else backend_ = hal::RelationBackend::AUTO;
        }
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[Relation] bad params for node {}: {}", node_id, e.what());
        return false;
    }
    return true;
}

bool RelationRecognitionNode::loadEngine() {
    engine_ = hal::RelationFactory::instance().create(backend_);
    if (!engine_) {
        LOG_ERROR("[Relation] no relation backend available "
                  "(check WITH_TENSORRT build option)");
        return false;
    }

    hal::RelationConfig hc;
    hc.model_path = cfg_.model_path;
    hc.predicate_bank_path = cfg_.predicate_bank_path;
    hc.img_size = cfg_.image_size > 0 ? cfg_.image_size : 448;
    hc.max_boxes = cfg_.max_boxes > 0 ? cfg_.max_boxes : 32;
    hc.batch_size = cfg_.batch_size;
    hc.precision = cfg_.precision;
    hc.predicates = cfg_.predicates;
    hc.per_predicate_threshold = cfg_.per_predicate_threshold;
    hc.score_threshold = cfg_.score_threshold;
    hc.max_relations = cfg_.max_relations;
    hc.device_id = cfg_.device_id;

    if (!engine_->loadModel(hc)) {
        LOG_ERROR_FMT("[Relation] failed to load model: {}", cfg_.model_path);
        engine_.reset();
        return false;
    }
    image_size_ = cfg_.image_size > 0 ? cfg_.image_size : engine_->getInputSize().first;
    LOG_INFO_FMT("[Relation] model loaded {} backend={} size={} predicates={} boundary={}",
                 cfg_.model_path, engine_->getBackendName(), image_size_,
                 cfg_.predicates.empty() ? "(engine default)" : std::to_string(cfg_.predicates.size()),
                 std::to_string(cfg_.boundary_classes.size()));
    return true;
}

bool RelationRecognitionNode::onStartup() {
    if (cfg_.model_path.empty()) {
        LOG_ERROR_FMT("[Relation] node {} has no model_path", node_id_);
        return false;
    }
    if (!loadEngine())
        return false;
    is_initialized_ = true;
    return true;
}

void RelationRecognitionNode::onShutdown() {
    std::lock_guard<std::mutex> lock(engine_mutex_);
    if (engine_) {
        engine_->unload();
        engine_.reset();
    }
    is_initialized_ = false;
}

cv::Mat RelationRecognitionNode::letterbox(const cv::Mat& src, float& scale,
                                           int& pad_x, int& pad_y) const {
    cv::Mat out(image_size_, image_size_, CV_8UC3, cv::Scalar(kLetterboxPad, kLetterboxPad, kLetterboxPad));
    if (src.empty())
        return out;
    const double r = std::min(static_cast<double>(image_size_) / src.rows,
                               static_cast<double>(image_size_) / src.cols);
    scale = static_cast<float>(r);
    const int nw = std::max(1, static_cast<int>(std::round(src.cols * r)));
    const int nh = std::max(1, static_cast<int>(std::round(src.rows * r)));
    pad_x = (image_size_ - nw) / 2;
    pad_y = (image_size_ - nh) / 2;
    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(out(cv::Rect(pad_x, pad_y, nw, nh)));
    return out;
}

void RelationRecognitionNode::fillTriplets(
        const core::InferenceResultPacket& in,
        const std::vector<hal::RelationTriplet>& triplets,
        const std::vector<std::string>& all_classes,
        const std::vector<int>& box_track_ids,
        std::vector<core::InferenceResultPacket::RelationResult>& out) {
    out.clear();
    for (const auto& t : triplets) {
        if (t.subject_index < 0 || t.object_index < 0)
            continue;
        if (static_cast<size_t>(t.subject_index) >= all_classes.size() ||
            static_cast<size_t>(t.object_index) >= all_classes.size())
            continue;
        core::InferenceResultPacket::RelationResult r;
        r.subject_class = all_classes[static_cast<size_t>(t.subject_index)];
        r.object_class = all_classes[static_cast<size_t>(t.object_index)];
        r.predicate = t.predicate;
        r.confidence = t.score;
        r.timestamp_ms = in.timestamp_ms;
        r.subject_track_id = static_cast<size_t>(t.subject_index) < box_track_ids.size()
                                 ? box_track_ids[static_cast<size_t>(t.subject_index)]
                                 : -1;
        r.object_track_id = static_cast<size_t>(t.object_index) < box_track_ids.size()
                                ? box_track_ids[static_cast<size_t>(t.object_index)]
                                : -1;
        out.push_back(std::move(r));
    }
}

void RelationRecognitionNode::processPacket(std::shared_ptr<core::BasePacket> packet) {
    if (!packet)
        return;
    if (packet->type == core::PacketType::STREAM_END) {
        broadcast(packet);
        return;
    }
    auto in = std::dynamic_pointer_cast<core::InferenceResultPacket>(packet);
    if (!in) {
        broadcast(packet);
        return;
    }
    if (!is_initialized_ || !engine_) {
        broadcast(packet);
        return;
    }
    in->relations.clear();
    // 隔帧降负载：非推理帧直接透传（relations 留空）。下游 child 规则靠
    // max_disappear_count 容忍这些空帧，不影响"持续贴墙"的判定。
    if (cfg_.run_every_n_frames > 1) {
        if ((++frame_counter_ % cfg_.run_every_n_frames) != 0) {
            broadcast(packet);
            return;
        }
    }
    // source_mat 必须有：detection_infer 已在原图坐标里给框，这里再把同一张
    // 原图 letterbox 到 448 去配它们。误用 mat（预处理后的 640）会让框和图
    // 不同尺寸，但不会报错，只表现为关系分莫名偏低。
    if (in->detections.empty() || !in->source_frame || !in->source_frame->source_mat) {
        if (!in->source_frame || !in->source_frame->source_mat)
            LOG_WARN_FMT("[Relation] frame {} has no source_mat, skipping", in->timestamp_ms);
        broadcast(packet);
        return;
    }

    // 1) 收候选框：过置信度门限，并把边界类别排在前面
    std::vector<const core::InferenceResultPacket::BBox*> picked;
    picked.reserve(in->detections.size());
    for (const auto& d : in->detections) {
        if (d.confidence >= cfg_.box_confidence)
            picked.push_back(&d);
    }
    if (picked.empty()) {
        broadcast(packet);
        return;
    }
    const size_t cap = cfg_.max_boxes > 0 ? static_cast<size_t>(cfg_.max_boxes)
                                          : picked.size();
    if (picked.size() > cap) {
        std::stable_sort(picked.begin(), picked.end(),
                         [](const core::InferenceResultPacket::BBox* a,
                            const core::InferenceResultPacket::BBox* b) {
                             return a->confidence > b->confidence;
                         });
        picked.resize(cap);
    }

    // 2) letterbox 到关系图边长（不能用上游 640 那份变换，两边比例不同）
    const cv::Mat& src = *in->source_frame->source_mat;
    float scale = 1.0f;
    int pad_x = 0, pad_y = 0;
    const cv::Mat lb = letterbox(src, scale, pad_x, pad_y);

    // 3) 框映射到 letterbox 空间——关系图要求框与 image 同一坐标系
    hal::RelationInput meta;
    meta.image = lb.data;
    meta.image_width = lb.cols;
    meta.image_height = lb.rows;
    meta.image_stride = static_cast<int>(lb.step);
    meta.is_bgr = true;
    std::vector<std::string> all_classes;
    std::vector<int> track_ids;
    meta.boxes.reserve(picked.size());
    for (const auto* d : picked) {
        hal::RelationInput::Box b;
        b.x1 = d->x * scale + pad_x;
        b.y1 = d->y * scale + pad_y;
        b.x2 = (d->x + d->w) * scale + pad_x;
        b.y2 = (d->y + d->h) * scale + pad_y;
        b.confidence = d->confidence;
        b.class_id = d->class_id;
        b.track_id = d->track_id;
        meta.boxes.push_back(b);
        all_classes.push_back(d->class_name);
        track_ids.push_back(d->track_id);
    }
    meta.class_names = all_classes;

    // 坐标系自检。框是原图坐标，映射后必须落在 448 画布内；若整体越界，
    // 说明 detections 已经在别的空间（比如预处理后的 640），这时继续跑只会
    // 得到一堆低分且无从察觉的假关系——直接停下来报错。
    if (!meta.boxes.empty()) {
        float max_x2 = 0.0f, max_y2 = 0.0f;
        for (const auto& b : meta.boxes) {
            max_x2 = std::max(max_x2, b.x2);
            max_y2 = std::max(max_y2, b.y2);
        }
        const bool inside = max_x2 <= static_cast<float>(image_size_) * 1.01f &&
                            max_y2 <= static_cast<float>(image_size_) * 1.01f;
        const bool all_zero = max_x2 <= 1.0f && max_y2 <= 1.0f;
        if (!inside || all_zero) {
            LOG_ERROR_FMT("[Relation] frame {} box coordinate space looks wrong "
                          "(max after remap = {:.1f},{:.1f}, canvas = {}); "
                          "detections are probably not in source-image coordinates",
                          in->timestamp_ms, max_x2, max_y2, image_size_);
            broadcast(packet);
            return;
        }
    }

    // 4) 推理
    std::vector<hal::RelationTriplet> triplets;
    const int64_t t0 = utils::TimeUtil::currentTimeMs();
    bool ok = false;
    {
        std::lock_guard<std::mutex> lock(engine_mutex_);
        ok = engine_ && engine_->infer(meta, triplets);
    }
    last_infer_ms_ = utils::TimeUtil::currentTimeMs() - t0;
    if (!ok) {
        LOG_WARN_FMT("[Relation] infer failed on frame {}", in->timestamp_ms);
        broadcast(packet);
        return;
    }

    std::vector<core::InferenceResultPacket::RelationResult> relations;
    fillTriplets(*in, triplets, all_classes, track_ids, relations);
    in->relations = std::move(relations);
    in->cost_time_map["relation_infer"] = last_infer_ms_;
    LOG_DEBUG_FMT("[Relation] frame {} boxes={} relations={} cost={}ms",
                  in->timestamp_ms, meta.boxes.size(), in->relations.size(), last_infer_ms_);
    // 有关系时才打 INFO：每帧都打会刷屏，而"一直没关系"正是需要排查的情况。
    if (!in->relations.empty()) {
        std::string s;
        for (const auto& r : in->relations) {
            s += fmt::format("{}(t{})->{}(t{}) {}={:.3f}; ",
                             r.subject_class.c_str(), r.subject_track_id,
                             r.object_class.c_str(), r.object_track_id,
                             r.predicate.c_str(), r.confidence);
        }
        LOG_INFO_FMT("[Relation] frame {} boxes={} -> {} relations: {}",
                     in->timestamp_ms, meta.boxes.size(), in->relations.size(), s);
    }
    broadcast(packet);
}

REGISTER_NODE("relation_recognition", RelationRecognitionNode)

} // namespace nodes
} // namespace ai_stream
