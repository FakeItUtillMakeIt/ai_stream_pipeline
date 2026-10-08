// src/hal/infer/nvidia/tensorrt_relation.cpp
// TensorRT 视觉关系引擎（RelateAnything 关系图）
#include "tensorrt_relation.h"
#include "ai_stream/hal/relation_factory.h"

#include "3rd_party/log_mgr/log_mgr.h"
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>

namespace ai_stream {
namespace hal {

namespace {

constexpr const char* kInImage = "image";
constexpr const char* kInBoxes = "boxes";
constexpr const char* kInCounts = "box_counts";
constexpr const char* kInW = "W";
constexpr const char* kInAlpha = "alpha";
constexpr const char* kOutPred = "pred_logits";
constexpr const char* kOutPair = "pair_logits";
constexpr const char* kOutSub = "sub_idx";
constexpr const char* kOutObj = "obj_idx";
constexpr const char* kOutValid = "valid_mask";

bool readJson(const std::string& path, nlohmann::json& out)
{
    std::ifstream in(path);
    if (!in) {
        LOG_WARN_FMT("[Relation] cannot open json: {}", path);
        return false;
    }
    try {
        in >> out;
    } catch (const std::exception& e) {
        LOG_WARN_FMT("[Relation] bad json {}: {}", path, e.what());
        return false;
    }
    return true;
}

} // namespace

TensorrtRelation::TensorrtRelation() = default;

TensorrtRelation::~TensorrtRelation() {
    unload();
}

bool TensorrtRelation::isAvailable() const {
    return true;
}

bool TensorrtRelation::loadModel(const RelationConfig& config) {
    config_ = config;
    if (config_.model_path.empty()) {
        LOG_ERROR("[Relation] model_path is empty");
        return false;
    }
    if (!core_.loadEngine(config_.model_path, "relation")) {
        LOG_ERROR_FMT("[Relation] failed to load engine {}", config_.model_path);
        return false;
    }

    std::string err;
    if (!config_.predicate_bank_path.empty()) {
        if (!loadPredicateBank(config_.predicate_bank_path, err)) {
            LOG_ERROR_FMT("[Relation] predicate bank failed: {}", err);
            unload();
            return false;
        }
    } else {
        LOG_WARN("[Relation] no predicate_bank_path; running with the engine's built-in vocabulary only");
    }

    // 标定参数来自 predicate bank（relateanything.json 的 calibration）
    thresholds_.pair_weight = 1.0f;
    thresholds_.calib_a = bank_.calib_a;
    thresholds_.calib_b = bank_.calib_b;
    thresholds_.threshold = config_.score_threshold;
    thresholds_.topk = config_.max_relations > 0 ? config_.max_relations : 20;
    thresholds_.max_per_pair = 1;
    thresholds_.box_score_weight = true;
    if (config_.img_size > 0)
        bank_.img_size = config_.img_size;

    // 谓词子集：配置优先，否则用引擎自带的默认集
    if (!config_.predicates.empty()) {
        if (!setPredicates(config_.predicates)) {
            LOG_ERROR_FMT("[Relation] predicates not found in bank: {}",
                          nlohmann::json(config_.predicates).dump());
            unload();
            return false;
        }
    } else if (!bank_.default_predicates.empty()) {
        setPredicates(bank_.default_predicates);
    } else if (!bank_.names.empty()) {
        // 没有默认集就把整张表都激活（V 会很大但不正确，只是别静默给空）
        setPredicates(bank_.names);
    }

    // 引擎按固定框数构建，先从引擎元数据读出来
    fixed_boxes_ = config_.max_boxes > 0 ? config_.max_boxes : 32;
    for (const auto& t : core_.tensors()) {
        if (t.name != kInBoxes || t.nb_dims < 2)
            continue;
        if (t.d[1] > 0)
            fixed_boxes_ = t.d[1];
    }
    fixed_predicates_ = static_cast<int64_t>(active_predicates_.size());

    if (!buildPredicateInputs(err)) {
        LOG_ERROR_FMT("[Relation] predicate input build failed: {}", err);
        unload();
        return false;
    }

    h_pred_.assign(static_cast<size_t>(fixed_boxes_) * static_cast<size_t>(fixed_predicates_), 0.0f);
    h_pair_.assign(static_cast<size_t>(fixed_boxes_), 0.0f);
    h_sub_.assign(static_cast<size_t>(fixed_boxes_), 0);
    h_obj_.assign(static_cast<size_t>(fixed_boxes_), 0);
    h_valid_.assign(static_cast<size_t>(fixed_boxes_), 0);

    loaded_ = true;
    LOG_INFO_FMT("[Relation] engine ready model={} boxes={} predicates={} size={}",
                config_.model_path, fixed_boxes_, fixed_predicates_, bank_.img_size);
    return true;
}

bool TensorrtRelation::loadPredicateBank(const std::string& path, std::string& err) {
    nlohmann::json j;
    if (!readJson(path, j)) {
        err = "cannot read " + path;
        return false;
    }
    bank_ = PredicateBank{};
    try {
        if (j.contains("predicates"))
            bank_.names = j["predicates"].get<std::vector<std::string>>();
        if (j.contains("default_predicates"))
            bank_.default_predicates = j["default_predicates"].get<std::vector<std::string>>();
        if (j.contains("img_size"))
            bank_.img_size = j["img_size"].get<int>();
        if (j.contains("text_dim"))
            bank_.text_dim = j["text_dim"].get<int>();
        if (j.contains("calibration")) {
            bank_.calib_a = j["calibration"].value("a", 1.0f);
            bank_.calib_b = j["calibration"].value("b", 0.0f);
        } else {
            bank_.calib_a = j.value("calib_a", 1.0f);
            bank_.calib_b = j.value("calib_b", 0.0f);
        }
        if (j.contains("W"))
            bank_.embed_W = j["W"].get<std::vector<float>>();
        if (j.contains("alpha"))
            bank_.embed_alpha = j["alpha"].get<std::vector<float>>();
        if (j.contains("thresholds"))
            bank_.thresholds = j["thresholds"].get<std::vector<float>>();
    } catch (const std::exception& e) {
        err = std::string("bad predicate bank schema: ") + e.what();
        return false;
    }

    if (bank_.names.empty()) {
        err = "predicate bank has no names";
        return false;
    }
    if (bank_.embed_alpha.empty() || bank_.embed_W.empty()) {
        err = "predicate bank has no W/alpha; the text encoder must run at export time";
        return false;
    }
    if (bank_.embed_alpha.size() != bank_.names.size()) {
        err = "alpha length != names length";
        return false;
    }
    const size_t dim = bank_.dim();
    if (bank_.embed_W.size() != dim * bank_.names.size()) {
        err = "W size mismatch: expect names*dim";
        return false;
    }
    return true;
}

bool TensorrtRelation::buildPredicateInputs(std::string& err) {
    if (bank_.empty()) {
        // 引擎内置谓词：W/alpha 是可选输入，不绑定
        return true;
    }
    const size_t dim = bank_.dim();
    h_w_.assign(active_predicates_.size() * dim, 0.0f);
    h_alpha_.assign(active_predicates_.size(), 0.0f);

    for (size_t i = 0; i < active_predicates_.size(); ++i) {
        const auto it = std::find(bank_.names.begin(), bank_.names.end(),
                                  active_predicates_[i]);
        if (it == bank_.names.end()) {
            err = "predicate not in bank: " + active_predicates_[i];
            return false;
        }
        const size_t row = static_cast<size_t>(std::distance(bank_.names.begin(), it));
        std::memcpy(h_w_.data() + i * dim, bank_.embed_W.data() + row * dim,
                    dim * sizeof(float));
        h_alpha_[i] = bank_.embed_alpha[row];
    }
    return true;
}

std::vector<std::string> TensorrtRelation::getPredicates() const {
    return active_predicates_;
}

bool TensorrtRelation::setPredicates(const std::vector<std::string>& predicates) {
    if (predicates.empty())
        return false;
    // 空表时不该再挂着旧谓词跑，否则阈值口径和配置对不上
    std::vector<std::string> previous = active_predicates_;
    active_predicates_ = predicates;
    std::string err;
    if (!buildPredicateInputs(err)) {
        active_predicates_ = std::move(previous);
        buildPredicateInputs(err);
        LOG_WARN_FMT("[Relation] setPredicates failed: {}", err);
        return false;
    }
    fixed_predicates_ = static_cast<int64_t>(active_predicates_.size());

    // 刻意**不**把 bank 里的逐谓词阈值灌进 per_predicate。
    // 那些是模型作者标定的工作点，实测关系分常在 0.4 量级，而 beside 的 bank
    // 阈值是 0.980——一旦让它覆盖配置的 score_threshold，规则就永远静默，
    // 而且不报任何错。per_predicate 留给"只压某个吵闹谓词"的场景，
    // 由 RelationConfig 显式传入，不自动生效。
    thresholds_.threshold = config_.score_threshold;
    thresholds_.per_predicate.clear();
    for (const auto& kv : config_.per_predicate_threshold)
        thresholds_.per_predicate[kv.first] = kv.second;
    return true;
}

std::pair<int, int> TensorrtRelation::getInputSize() const {
    return {bank_.img_size, bank_.img_size};
}

float TensorrtRelation::getScoreThreshold() const {
    return thresholds_.threshold;
}

bool TensorrtRelation::setScoreThreshold(float threshold) {
    thresholds_.threshold = threshold;
    config_.score_threshold = threshold;
    return true;
}

bool TensorrtRelation::uploadImage(const RelationInput& input) {
    const int s = bank_.img_size;
    if (input.image == nullptr || input.image_width != s || input.image_height != s) {
        LOG_WARN_FMT("[Relation] image must be letterboxed to {}x{}, got {}x{}",
                     s, s, input.image_width, input.image_height);
        return false;
    }
    const int stride = input.image_stride > 0 ? input.image_stride : input.image_width * 3;
    // HWC uint8 -> NCHW float 0..1；DINOv3 用 ImageNet 归一化，但关系图的 ONNX
    // 导出里已经含了归一化，这里只做 /255，保持与 deploy/runtime.py 一致。
    const size_t plane = static_cast<size_t>(s) * s;
    std::vector<float> nchw(plane * 3);
    for (int y = 0; y < s; ++y) {
        const uint8_t* row = input.image + static_cast<size_t>(y) * stride;
        for (int x = 0; x < s; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * 3;
            if (input.is_bgr) {
                nchw[0 * plane + y * s + x] = px[2] / 255.0f;  // R
                nchw[1 * plane + y * s + x] = px[1] / 255.0f;  // G
                nchw[2 * plane + y * s + x] = px[0] / 255.0f;  // B
            } else {
                nchw[0 * plane + y * s + x] = px[0] / 255.0f;
                nchw[1 * plane + y * s + x] = px[1] / 255.0f;
                nchw[2 * plane + y * s + x] = px[2] / 255.0f;
            }
        }
    }
    void* dst = core_.buffer(kInImage);
    if (dst == nullptr)
        return false;
    auto stream = static_cast<cudaStream_t>(core_.stream());
    cudaError_t err = cudaMemcpyAsync(dst, nchw.data(), nchw.size() * sizeof(float),
                                      cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("[Relation] image H2D failed: {}", cudaGetErrorString(err));
        return false;
    }
    return true;
}

bool TensorrtRelation::runGraph() {
    return core_.enqueue() && core_.synchronize();
}

void TensorrtRelation::copyOutputsToHost() {
    auto pull = [&](const char* name, void* host, size_t bytes) {
        void* src = core_.buffer(name);
        if (src == nullptr)
            return;
        cudaMemcpyAsync(host, src, bytes, cudaMemcpyDeviceToHost,
                        static_cast<cudaStream_t>(core_.stream()));
    };
    pull(kOutPred, h_pred_.data(), h_pred_.size() * sizeof(float));
    pull(kOutPair, h_pair_.data(), h_pair_.size() * sizeof(float));
    pull(kOutSub, h_sub_.data(), h_sub_.size() * sizeof(int32_t));
    pull(kOutObj, h_obj_.data(), h_obj_.size() * sizeof(int32_t));
    pull(kOutValid, h_valid_.data(), h_valid_.size() * sizeof(uint8_t));
}

bool TensorrtRelation::infer(const RelationInput& input,
                             std::vector<RelationTriplet>& out) {
    out.clear();
    if (!uploadImage(input))
        return false;
    return runBoxes(input, out);
}

bool TensorrtRelation::runBoxes(const RelationInput& input,
                                std::vector<RelationTriplet>& out) {
    out.clear();
    const int n = static_cast<int>(input.boxes.size());
    if (n == 0)
        return true;                 // 没框是正常情况，不是失败

    const int64_t k = fixed_boxes_;
    const int64_t v = fixed_predicates_;
    if (v <= 0) {
        LOG_ERROR("[Relation] no active predicate; the graph would score nothing meaningful");
        return false;
    }

    // 图按固定 N 建，多余位置零填充——这也是 decode 侧要按 in_range 过滤的原因
    if (n > k) {
        LOG_DEBUG_FMT("[Relation] {} boxes exceed engine capacity {}, truncating", n, k);
    }
    const int64_t n_used = std::min<int64_t>(n, k);

    // 形状：image/boxes 随谓词数与框数变
    {
        int64_t d[4] = {1, 3, bank_.img_size, bank_.img_size};
        if (!core_.setInputShape(kInImage, d, 4)) return false;
        int64_t b[3] = {1, k, 4};
        if (!core_.setInputShape(kInBoxes, b, 3)) return false;
        int64_t c[2] = {1, 1};
        if (!core_.setInputShape(kInCounts, c, 2)) return false;
        if (!h_w_.empty()) {
            int64_t w[2] = {v, static_cast<int64_t>(bank_.dim())};
            if (!core_.setInputShape(kInW, w, 2)) return false;
            int64_t a[2] = {v, 1};
            if (!core_.setInputShape(kInAlpha, a, 2)) return false;
        }
    }

    // 框：xyxy，且必须与 image 在同一个 letterbox 空间
    std::vector<float> boxes(static_cast<size_t>(k) * 4, 0.0f);
    box_scores_.assign(static_cast<size_t>(n_used), 0.0f);
    for (int64_t i = 0; i < n_used; ++i) {
        const auto& b = input.boxes[static_cast<size_t>(i)];
        boxes[static_cast<size_t>(i) * 4 + 0] = b.x1;
        boxes[static_cast<size_t>(i) * 4 + 1] = b.y1;
        boxes[static_cast<size_t>(i) * 4 + 2] = b.x2;
        boxes[static_cast<size_t>(i) * 4 + 3] = b.y2;
        box_scores_[static_cast<size_t>(i)] = b.confidence;
    }

    void* d_boxes = core_.buffer(kInBoxes);
    void* d_counts = core_.buffer(kInCounts);
    if (d_boxes == nullptr || d_counts == nullptr) return false;
    cudaMemcpyAsync(d_boxes, boxes.data(), boxes.size() * sizeof(float),
                    cudaMemcpyHostToDevice, static_cast<cudaStream_t>(core_.stream()));
    const int32_t count32 = static_cast<int32_t>(n_used);
    cudaMemcpyAsync(d_counts, &count32, sizeof(int32_t),
                    cudaMemcpyHostToDevice, static_cast<cudaStream_t>(core_.stream()));
    if (!h_w_.empty()) {
        void* d_w = core_.buffer(kInW);
        void* d_a = core_.buffer(kInAlpha);
        if (d_w) cudaMemcpyAsync(d_w, h_w_.data(), h_w_.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, static_cast<cudaStream_t>(core_.stream()));
        if (d_a) cudaMemcpyAsync(d_a, h_alpha_.data(), h_alpha_.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, static_cast<cudaStream_t>(core_.stream()));
    }

    if (!runGraph()) {
        LOG_WARN("[Relation] graph execution failed");
        return false;
    }
    copyOutputsToHost();
    core_.synchronize();

    // 只取前 v 个谓词列：图可能仍按完整词表算，尾部是无关列
    std::vector<float> pred(static_cast<size_t>(k) * static_cast<size_t>(v));
    for (int64_t i = 0; i < k; ++i)
        for (int64_t j = 0; j < v; ++j)
            pred[static_cast<size_t>(i * v + j)] = h_pred_[static_cast<size_t>(i) * h_pred_.size() / k + j];

    std::vector<DecodedRelation> decoded;
    decodeRelations(pred, h_pair_, h_sub_, h_obj_, h_valid_, active_predicates_,
                    thresholds_, box_scores_, decoded);
    const size_t cap = config_.max_relations > 0
                           ? static_cast<size_t>(config_.max_relations)
                           : decoded.size();
    out.reserve(std::min(cap, decoded.size()));
    for (size_t i = 0; i < decoded.size() && out.size() < cap; ++i) {
        RelationTriplet t;
        t.subject_index = decoded[i].subject_index;
        t.object_index = decoded[i].object_index;
        t.predicate = decoded[i].predicate;
        t.score = decoded[i].score;
        t.predicate_index = decoded[i].predicate_index;
        out.push_back(std::move(t));
    }
    return true;
}

bool TensorrtRelation::inferPreprocessed(const float* input_nchw, size_t size_bytes,
                                         const RelationInput& meta,
                                         std::vector<RelationTriplet>& out) {
    out.clear();
    if (!loaded_ || input_nchw == nullptr) {
        LOG_WARN("[Relation] inferPreprocessed before loadModel or with null input");
        return false;
    }
    void* dst = core_.buffer(kInImage);
    if (dst == nullptr)
        return false;
    cudaError_t err = cudaMemcpyAsync(dst, input_nchw, size_bytes,
                                      cudaMemcpyHostToDevice,
                                      static_cast<cudaStream_t>(core_.stream()));
    if (err != cudaSuccess) {
        LOG_ERROR_FMT("[Relation] preprocessed image H2D failed: {}", cudaGetErrorString(err));
        return false;
    }
    return runBoxes(meta, out);
}

void TensorrtRelation::unload() {
    loaded_ = false;
    active_predicates_.clear();
    h_pred_.clear(); h_pair_.clear(); h_sub_.clear(); h_obj_.clear(); h_valid_.clear();
    h_w_.clear(); h_alpha_.clear(); box_scores_.clear();
    bank_ = PredicateBank{};
}

REGISTER_RELATION_BACKEND(RelationBackend::TENSORRT, TensorrtRelation)

} // namespace hal
} // namespace ai_stream
