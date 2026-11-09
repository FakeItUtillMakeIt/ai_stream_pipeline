// src/hal/infer/ascend/ascend_relation.cpp
#include "ascend_relation.h"
#include "ai_stream/hal/relation_factory.h"
#include "ascend/ascend_acl.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstring>
#include <fstream>
#include <unistd.h>
#include <vector>

namespace ai_stream {
namespace hal {
namespace {

bool readJsonFile(const std::string& path, nlohmann::json& out) {
    std::ifstream f(path);
    if (!f.is_open()) return false;
    try {
        f >> out;
        return true;
    } catch (...) {
        return false;
    }
}

size_t elementsFromDims(const aclmdlIODims& dims) {
    if (dims.dimCount == 0) return 1;
    size_t n = 1;
    for (size_t i = 0; i < dims.dimCount; ++i) {
        if (dims.dims[i] <= 0) return 0;
        n *= static_cast<size_t>(dims.dims[i]);
    }
    return n;
}

} // namespace

AscendRelation::~AscendRelation() {
    cleanup();
}

bool AscendRelation::isAvailable() const {
    return ::access("/dev/davinci0", R_OK | W_OK) == 0;
}

std::string AscendRelation::getBackendName() const {
    return soc_name_.empty() ? "Ascend ACL Relation (Ascend NPU)"
                             : ("Ascend ACL Relation (" + soc_name_ + ")");
}

bool AscendRelation::initAcl() {
    if (!ensureAscendAclInitialized()) {
        LOG_ERROR("[AscendRelation] aclInit failed");
        return false;
    }
    uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS || count == 0) {
        LOG_ERROR("[AscendRelation] No Ascend device found");
        return false;
    }
    int device = config_.device_id >= 0 ? config_.device_id : 0;
    if (static_cast<uint32_t>(device) >= count) device = 0;
    if (aclrtSetDevice(device) != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendRelation] aclrtSetDevice({}) failed", device);
        return false;
    }
    config_.device_id = device;
    const char* soc = aclrtGetSocName();
    if (soc) soc_name_ = soc;
    return true;
}

bool AscendRelation::loadModel(const RelationConfig& config) {
    cleanup();
    config_ = config;

    if (!initAcl()) {
        cleanup();
        return false;
    }

    aclError ret = aclmdlLoadFromFile(config.model_path.c_str(), &model_id_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendRelation] aclmdlLoadFromFile failed: {} ({}); only .om is supported",
                      static_cast<int>(ret), config.model_path);
        cleanup();
        return false;
    }
    loaded_ = true;

    desc_ = aclmdlCreateDesc();
    if (!desc_ || aclmdlGetDesc(desc_, model_id_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendRelation] aclmdlGetDesc failed");
        cleanup();
        return false;
    }

    std::string err;
    if (config_.predicate_bank_path.empty() ||
        !loadPredicateBank(config_.predicate_bank_path, err)) {
        LOG_ERROR_FMT("[AscendRelation] predicate bank failed: {}",
                      config_.predicate_bank_path.empty() ? "path is empty" : err);
        cleanup();
        return false;
    }
    if (!parseIo()) {
        cleanup();
        return false;
    }
    if (static_cast<int64_t>(bank_.names.size()) != fixed_predicates_) {
        LOG_ERROR_FMT("[AscendRelation] bank predicates={} but model expects {}",
                      bank_.names.size(), fixed_predicates_);
        cleanup();
        return false;
    }

    thresholds_.pair_weight = 1.0f;
    thresholds_.calib_a = bank_.calib_a;
    thresholds_.calib_b = bank_.calib_b;
    thresholds_.threshold = config_.score_threshold;
    thresholds_.topk = config_.max_relations > 0 ? config_.max_relations : 20;
    thresholds_.max_per_pair = 1;
    thresholds_.box_score_weight = true;
    thresholds_.per_predicate.clear();
    for (const auto& kv : config_.per_predicate_threshold) {
        thresholds_.per_predicate[kv.first] = kv.second;
    }

    if (!config_.predicates.empty()) {
        active_predicates_ = config_.predicates;
    } else if (!bank_.default_predicates.empty()) {
        active_predicates_ = bank_.default_predicates;
    } else {
        active_predicates_ = bank_.names;
    }
    if (!buildPredicateInputs(err) || !uploadPredicateInputs()) {
        LOG_ERROR_FMT("[AscendRelation] build/upload predicate inputs failed: {}", err);
        cleanup();
        return false;
    }

    h_pred_.assign(static_cast<size_t>(pairs_ * fixed_predicates_), 0.0f);
    h_pair_.assign(static_cast<size_t>(pairs_), 0.0f);
    h_sub_.assign(static_cast<size_t>(pairs_), 0);
    h_obj_.assign(static_cast<size_t>(pairs_), 0);
    h_valid_.assign(static_cast<size_t>(pairs_), 0);

    LOG_INFO_FMT("[AscendRelation] Loaded '{}' on {} (boxes={}, pairs={}, predicates={}, size={})",
                 config.model_path, getBackendName(), fixed_boxes_, pairs_,
                 fixed_predicates_, bank_.img_size);
    return true;
}

bool AscendRelation::loadPredicateBank(const std::string& path, std::string& err) {
    nlohmann::json j;
    if (!readJsonFile(path, j)) {
        err = "cannot read " + path;
        return false;
    }
    bank_ = RelPredicateBank{};
    try {
        if (j.contains("predicates")) bank_.names = j["predicates"].get<std::vector<std::string>>();
        if (j.contains("default_predicates"))
            bank_.default_predicates = j["default_predicates"].get<std::vector<std::string>>();
        if (j.contains("img_size")) bank_.img_size = j["img_size"].get<int>();
        if (j.contains("text_dim")) bank_.text_dim = j["text_dim"].get<int>();
        if (j.contains("calibration")) {
            bank_.calib_a = j["calibration"].value("a", 1.0f);
            bank_.calib_b = j["calibration"].value("b", 0.0f);
        } else {
            bank_.calib_a = j.value("calib_a", 1.0f);
            bank_.calib_b = j.value("calib_b", 0.0f);
        }
        if (j.contains("W")) bank_.embed_W = j["W"].get<std::vector<float>>();
        if (j.contains("alpha")) bank_.embed_alpha = j["alpha"].get<std::vector<float>>();
        if (j.contains("thresholds"))
            bank_.thresholds = j["thresholds"].get<std::vector<float>>();
    } catch (const std::exception& e) {
        err = std::string("bad predicate bank schema: ") + e.what();
        return false;
    }
    if (bank_.names.empty()) { err = "bank has no names"; return false; }
    if (bank_.embed_alpha.empty() || bank_.embed_W.empty()) {
        err = "bank has no W/alpha";
        return false;
    }
    if (bank_.embed_alpha.size() != bank_.names.size()) {
        err = "alpha length != names length";
        return false;
    }
    const size_t dim = bank_.dim();
    if (dim == 0 || bank_.embed_W.size() != dim * bank_.names.size()) {
        err = "W size mismatch";
        return false;
    }
    if (bank_.text_dim > 0 && static_cast<size_t>(bank_.text_dim) != dim) {
        err = "text_dim disagrees with W/alpha implied dim";
        return false;
    }
    if (config_.img_size > 0) bank_.img_size = config_.img_size;
    return true;
}

bool AscendRelation::parseIo() {
    const size_t in_count = aclmdlGetNumInputs(desc_);
    const size_t out_count = aclmdlGetNumOutputs(desc_);
    // 兼容两种 OM：
    //   5 输入：image/boxes/box_counts/W/alpha（原始动态图）
    //   4 输入：box_counts 已按 32 固化成常量（If/RoPE 静态化的版本）
    if (in_count < 4 || out_count < 5) {
        LOG_ERROR_FMT("[AscendRelation] Expect >=4 in/out, got {} / {}", in_count, out_count);
        return false;
    }

    d_in_.assign(in_count, nullptr);
    in_size_.assign(in_count, 0);
    in_type_.assign(in_count, ACL_FLOAT);
    in_buf_.assign(in_count, nullptr);
    d_out_.assign(out_count, nullptr);
    out_size_.assign(out_count, 0);
    out_type_.assign(out_count, ACL_FLOAT);
    out_buf_.assign(out_count, nullptr);

    for (size_t i = 0; i < in_count; ++i) {
        const char* name = aclmdlGetInputNameByIndex(desc_, i);
        const std::string n = name ? name : "";
        if (n.find("image") != std::string::npos) in_image_ = static_cast<int>(i);
        else if (n.find("boxes") != std::string::npos) in_boxes_ = static_cast<int>(i);
        else if (n.find("box_counts") != std::string::npos) in_counts_ = static_cast<int>(i);
        else if (n == "W" || n.find("W") != std::string::npos) in_w_ = static_cast<int>(i);
        else if (n.find("alpha") != std::string::npos) in_alpha_ = static_cast<int>(i);
    }
    if (in_image_ < 0 || in_boxes_ < 0 || in_w_ < 0 || in_alpha_ < 0) {
        LOG_ERROR("[AscendRelation] Failed to map image/boxes/W/alpha inputs by name");
        return false;
    }

    aclmdlIODims image_dims{};
    aclmdlGetInputDims(desc_, in_image_, &image_dims);
    if (image_dims.dimCount == 4 && image_dims.dims[2] > 0) {
        img_size_ = static_cast<int>(image_dims.dims[2]);
    }
    aclmdlIODims boxes_dims{};
    aclmdlGetInputDims(desc_, in_boxes_, &boxes_dims);
    if (boxes_dims.dimCount == 3 && boxes_dims.dims[1] > 0) {
        fixed_boxes_ = boxes_dims.dims[1];
    }

    for (size_t i = 0; i < in_count; ++i) {
        in_size_[i] = aclmdlGetInputSizeByIndex(desc_, i);
        in_type_[i] = aclmdlGetInputDataType(desc_, i);
        if (in_size_[i] == 0) {
            LOG_ERROR_FMT("[AscendRelation] input[{}] size is 0", i);
            return false;
        }
        if (aclrtMalloc(&d_in_[i], in_size_[i], ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            !d_in_[i]) {
            LOG_ERROR_FMT("[AscendRelation] aclrtMalloc input[{}] failed", i);
            return false;
        }
        in_buf_[i] = aclCreateDataBuffer(d_in_[i], in_size_[i]);
    }
    input_ds_ = aclmdlCreateDataset();
    if (!input_ds_) return false;
    for (size_t i = 0; i < in_count; ++i) {
        if (!in_buf_[i] ||
            aclmdlAddDatasetBuffer(input_ds_, in_buf_[i]) != ACL_SUCCESS) {
            LOG_ERROR("[AscendRelation] add input dataset buffer failed");
            return false;
        }
    }

    output_ds_ = aclmdlCreateDataset();
    if (!output_ds_) return false;
    for (size_t i = 0; i < out_count; ++i) {
        const char* name = aclmdlGetOutputNameByIndex(desc_, i);
        const std::string n = name ? name : "";
        if (n.find("pred_logits") != std::string::npos) out_pred_ = static_cast<int>(i);
        else if (n.find("pair_logits") != std::string::npos) out_pair_ = static_cast<int>(i);
        else if (n.find("sub_idx") != std::string::npos) out_sub_ = static_cast<int>(i);
        else if (n.find("obj_idx") != std::string::npos) out_obj_ = static_cast<int>(i);
        else if (n.find("valid_mask") != std::string::npos) out_valid_ = static_cast<int>(i);

        out_size_[i] = aclmdlGetOutputSizeByIndex(desc_, i);
        out_type_[i] = aclmdlGetOutputDataType(desc_, i);
        if (out_size_[i] == 0) {
            LOG_ERROR_FMT("[AscendRelation] output[{}] size is 0", i);
            return false;
        }
        if (aclrtMalloc(&d_out_[i], out_size_[i], ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
            !d_out_[i]) {
            LOG_ERROR_FMT("[AscendRelation] aclrtMalloc output[{}] failed", i);
            return false;
        }
        out_buf_[i] = aclCreateDataBuffer(d_out_[i], out_size_[i]);
        if (!out_buf_[i] ||
            aclmdlAddDatasetBuffer(output_ds_, out_buf_[i]) != ACL_SUCCESS) {
            LOG_ERROR("[AscendRelation] add output dataset buffer failed");
            return false;
        }
    }
    if (out_pred_ < 0 || out_pair_ < 0 || out_sub_ < 0 || out_obj_ < 0 || out_valid_ < 0) {
        LOG_ERROR("[AscendRelation] Failed to map 5 outputs by name");
        return false;
    }

    LOG_INFO_FMT("[AscendRelation] inputs image={} boxes={} counts={} W={} alpha={}",
                 in_image_, in_boxes_, in_counts_, in_w_, in_alpha_);
    aclmdlIODims pred_dims{};
    if (aclmdlGetOutputDims(desc_, out_pred_, &pred_dims) == ACL_SUCCESS && pred_dims.dimCount == 3) {
        pairs_ = pred_dims.dims[1] > 0 ? pred_dims.dims[1] : 0;
        fixed_predicates_ = pred_dims.dims[2] > 0 ? pred_dims.dims[2] : 0;
    }
    if (pairs_ <= 0 || fixed_predicates_ <= 0) {
        LOG_ERROR("[AscendRelation] Cannot read pairs/predicates from pred_logits dims");
        return false;
    }
    const size_t need_pred = static_cast<size_t>(pairs_ * fixed_predicates_) * sizeof(float);
    if (out_size_[out_pred_] < need_pred) {
        LOG_ERROR_FMT("[AscendRelation] pred_logits size {} < {}", out_size_[out_pred_], need_pred);
        return false;
    }
    return true;
}

bool AscendRelation::buildPredicateInputs(std::string& err) {
    const size_t dim = bank_.dim();
    const size_t vocab = bank_.names.size();
    h_w_.assign(vocab * dim, 0.0f);
    h_alpha_.assign(vocab, 0.0f);
    for (const auto& name : active_predicates_) {
        auto it = std::find(bank_.names.begin(), bank_.names.end(), name);
        if (it == bank_.names.end()) {
            err = "predicate not in bank: " + name;
            return false;
        }
        const size_t row = static_cast<size_t>(std::distance(bank_.names.begin(), it));
        std::memcpy(h_w_.data() + row * dim, bank_.embed_W.data() + row * dim,
                    dim * sizeof(float));
        h_alpha_[row] = bank_.embed_alpha[row];
    }
    return true;
}

bool AscendRelation::uploadPredicateInputs() {
    if (h_w_.empty() || h_alpha_.empty()) return false;
    const size_t w_bytes = h_w_.size() * sizeof(float);
    const size_t a_bytes = h_alpha_.size() * sizeof(float);
    if (in_size_[in_w_] < w_bytes || in_size_[in_alpha_] < a_bytes) {
        LOG_ERROR("[AscendRelation] W/alpha input buffer smaller than bank");
        return false;
    }
    if (aclrtMemcpy(d_in_[in_w_], in_size_[in_w_], h_w_.data(), w_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(d_in_[in_alpha_], in_size_[in_alpha_], h_alpha_.data(), a_bytes,
                    ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        LOG_ERROR("[AscendRelation] H2D W/alpha failed");
        return false;
    }
    return true;
}

bool AscendRelation::setPredicates(const std::vector<std::string>& predicates) {
    if (predicates.empty()) return false;
    std::vector<std::string> previous = active_predicates_;
    active_predicates_ = predicates;
    std::string err;
    if (!buildPredicateInputs(err) || !uploadPredicateInputs()) {
        active_predicates_ = std::move(previous);
        buildPredicateInputs(err);
        uploadPredicateInputs();
        LOG_WARN_FMT("[AscendRelation] setPredicates failed: {}", err);
        return false;
    }
    thresholds_.threshold = config_.score_threshold;
    thresholds_.per_predicate.clear();
    for (const auto& kv : config_.per_predicate_threshold) {
        thresholds_.per_predicate[kv.first] = kv.second;
    }
    return true;
}

bool AscendRelation::setScoreThreshold(float threshold) {
    thresholds_.threshold = threshold;
    config_.score_threshold = threshold;
    return true;
}

bool AscendRelation::infer(const RelationInput& input,
                           std::vector<RelationTriplet>& out) {
    out.clear();
    if (!loaded_ || !input.image) return false;
    if (input.image_width != img_size_ || input.image_height != img_size_) {
        LOG_WARN_FMT("[AscendRelation] image must be {}x{}, got {}x{}",
                     img_size_, img_size_, input.image_width, input.image_height);
        return false;
    }
    const int n = static_cast<int>(input.boxes.size());
    if (n == 0) return true;
    if (aclrtSetDevice(config_.device_id) != ACL_SUCCESS) {
        LOG_ERROR("[AscendRelation] set device failed");
        return false;
    }

    const int64_t n_actual = std::min<int64_t>(n, fixed_boxes_);
    // 4 输入 OM 把 box_counts 固化成 32，必须送满 32 个框；5 输入 OM 用真实框数
    const int64_t n_graph = (in_counts_ >= 0) ? n_actual : fixed_boxes_;

    // image: HWC uint8 -> NCHW float /255
    const int stride = input.image_stride > 0 ? input.image_stride : input.image_width * 3;
    const size_t plane = static_cast<size_t>(img_size_) * img_size_;
    std::vector<float> nchw(plane * 3);
    for (int y = 0; y < img_size_; ++y) {
        const uint8_t* row = input.image + static_cast<size_t>(y) * stride;
        for (int x = 0; x < img_size_; ++x) {
            const uint8_t* px = row + static_cast<size_t>(x) * 3;
            const size_t idx = static_cast<size_t>(y) * img_size_ + x;
            if (input.is_bgr) {
                nchw[0 * plane + idx] = px[2] / 255.0f;
                nchw[1 * plane + idx] = px[1] / 255.0f;
                nchw[2 * plane + idx] = px[0] / 255.0f;
            } else {
                nchw[0 * plane + idx] = px[0] / 255.0f;
                nchw[1 * plane + idx] = px[1] / 255.0f;
                nchw[2 * plane + idx] = px[2] / 255.0f;
            }
        }
    }
    if (aclrtMemcpy(d_in_[in_image_], in_size_[in_image_], nchw.data(),
                    nchw.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        LOG_ERROR("[AscendRelation] H2D image failed");
        return false;
    }

    // boxes: normalized cxcywh in [0,1], zero padded to fixed_boxes_
    std::vector<float> boxes(static_cast<size_t>(fixed_boxes_) * 4, 0.0f);
    box_scores_.assign(static_cast<size_t>(n_graph), 0.0f);
    const float side = static_cast<float>(img_size_);
    for (int64_t i = 0; i < n_actual; ++i) {
        const auto& b = input.boxes[static_cast<size_t>(i)];
        const float x1 = b.x1 / side, y1 = b.y1 / side;
        const float x2 = b.x2 / side, y2 = b.y2 / side;
        boxes[static_cast<size_t>(i) * 4 + 0] = (x1 + x2) / 2.0f;
        boxes[static_cast<size_t>(i) * 4 + 1] = (y1 + y2) / 2.0f;
        boxes[static_cast<size_t>(i) * 4 + 2] = (x2 - x1);
        boxes[static_cast<size_t>(i) * 4 + 3] = (y2 - y1);
        box_scores_[static_cast<size_t>(i)] = b.confidence;
    }
    if (aclrtMemcpy(d_in_[in_boxes_], in_size_[in_boxes_], boxes.data(),
                    boxes.size() * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
        LOG_ERROR("[AscendRelation] H2D boxes failed");
        return false;
    }

    // box_counts 仅 5 输入 OM 才有；4 输入 OM 内部常量 32
    if (in_counts_ >= 0) {
        if (in_type_[in_counts_] == ACL_INT64) {
            const int64_t count = n_graph;
            if (aclrtMemcpy(d_in_[in_counts_], in_size_[in_counts_], &count, sizeof(count),
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                LOG_ERROR("[AscendRelation] H2D counts failed");
                return false;
            }
        } else {
            const int32_t count = static_cast<int32_t>(n_graph);
            if (aclrtMemcpy(d_in_[in_counts_], in_size_[in_counts_], &count, sizeof(count),
                            ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
                LOG_ERROR("[AscendRelation] H2D counts failed");
                return false;
            }
        }
    }

    aclError ret = aclmdlExecute(model_id_, input_ds_, output_ds_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendRelation] aclmdlExecute failed: {}", static_cast<int>(ret));
        return false;
    }

    const size_t pred_bytes = h_pred_.size() * sizeof(float);
    const size_t pair_bytes = h_pair_.size() * sizeof(float);
    const size_t sub_bytes = h_sub_.size() * sizeof(int64_t);
    const size_t valid_bytes = h_valid_.size() * sizeof(uint8_t);
    if (aclrtMemcpy(h_pred_.data(), std::min(pred_bytes, out_size_[out_pred_]), d_out_[out_pred_],
                    std::min(pred_bytes, out_size_[out_pred_]), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(h_pair_.data(), std::min(pair_bytes, out_size_[out_pair_]), d_out_[out_pair_],
                    std::min(pair_bytes, out_size_[out_pair_]), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(h_sub_.data(), std::min(sub_bytes, out_size_[out_sub_]), d_out_[out_sub_],
                    std::min(sub_bytes, out_size_[out_sub_]), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(h_obj_.data(), std::min(sub_bytes, out_size_[out_obj_]), d_out_[out_obj_],
                    std::min(sub_bytes, out_size_[out_obj_]), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS ||
        aclrtMemcpy(h_valid_.data(), std::min(valid_bytes, out_size_[out_valid_]), d_out_[out_valid_],
                    std::min(valid_bytes, out_size_[out_valid_]), ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
        LOG_ERROR("[AscendRelation] D2H outputs failed");
        return false;
    }

    std::vector<DecodedRelation> decoded;
    decodeRelations(h_pred_, h_pair_, h_sub_, h_obj_, h_valid_, bank_.names,
                    thresholds_, box_scores_, decoded, active_predicates_);
    const size_t cap = config_.max_relations > 0
                           ? static_cast<size_t>(config_.max_relations)
                           : decoded.size();
    out.reserve(std::min(cap, decoded.size()));
    for (size_t i = 0; i < decoded.size() && out.size() < cap; ++i) {
        // 4 输入 OM 的 padding 框（索引 >= n_actual）会产生无效配对，host 过滤
        if (decoded[i].subject_index >= n_actual ||
            decoded[i].object_index >= n_actual) {
            continue;
        }
        RelationTriplet t;
        t.subject_index = decoded[i].subject_index;
        t.object_index = decoded[i].object_index;
        t.predicate = decoded[i].predicate;
        t.score = decoded[i].score;
        t.predicate_index = decoded[i].predicate_index;
        out.push_back(std::move(t));
    }

    size_t valid_cnt = 0;
    for (uint8_t v : h_valid_) valid_cnt += (v != 0) ? 1u : 0u;
    LOG_DEBUG_FMT("[AscendRelation] pairs={} valid={} boxes_actual={} graph_boxes={} relations={}",
                  h_valid_.size(), valid_cnt, n_actual, n_graph, out.size());
    return true;
}

void AscendRelation::unload() {
    cleanup();
}

void AscendRelation::cleanup() {
    if (input_ds_) {
        aclmdlDestroyDataset(input_ds_);
        input_ds_ = nullptr;
    }
    for (auto* b : in_buf_) if (b) aclDestroyDataBuffer(b);
    in_buf_.clear();
    for (auto* p : d_in_) if (p) aclrtFree(p);
    d_in_.clear();
    in_size_.clear();
    in_type_.clear();

    if (output_ds_) {
        aclmdlDestroyDataset(output_ds_);
        output_ds_ = nullptr;
    }
    for (auto* b : out_buf_) if (b) aclDestroyDataBuffer(b);
    out_buf_.clear();
    for (auto* p : d_out_) if (p) aclrtFree(p);
    d_out_.clear();
    out_size_.clear();
    out_type_.clear();

    if (desc_) {
        aclmdlDestroyDesc(desc_);
        desc_ = nullptr;
    }
    if (loaded_) {
        aclmdlUnload(model_id_);
        loaded_ = false;
    }
    in_image_ = in_boxes_ = in_counts_ = in_w_ = in_alpha_ = -1;
    out_pred_ = out_pair_ = out_sub_ = out_obj_ = out_valid_ = -1;
    h_w_.clear();
    h_alpha_.clear();
    h_pred_.clear();
    h_pair_.clear();
    h_sub_.clear();
    h_obj_.clear();
    h_valid_.clear();
    box_scores_.clear();
}

REGISTER_RELATION_BACKEND(RelationBackend::ASCEND, AscendRelation)

} // namespace hal
} // namespace ai_stream
