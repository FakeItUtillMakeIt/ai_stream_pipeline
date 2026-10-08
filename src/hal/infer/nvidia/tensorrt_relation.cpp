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
    // 标定参数来自 predicate bank。默认值 (1,0) 表示未标定，此时分数与
    // bank 的标定口径差一个 sigmoid 变换，阈值全线失效且不报错——这是最
    // 容易被误判成"模型效果不好"而查错方向的情况，所以显式打出来。
    LOG_INFO_FMT("[Relation] calibration a={} b={} threshold={} topk={}",
                 thresholds_.calib_a, thresholds_.calib_b,
                 thresholds_.threshold, thresholds_.topk);
    if (bank_.calib_a == 1.0f && bank_.calib_b == 0.0f)
        LOG_WARN("[Relation] predicate bank carried no calibration; scores are "
                 "uncalibrated raw probabilities, thresholds will not transfer");
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
    // 输出 pred_logits 的最后一维是整张表（静态 243），不是激活数
    fixed_predicates_ = bank_.empty()
                            ? static_cast<int64_t>(active_predicates_.size())
                            : static_cast<int64_t>(bank_.names.size());

    // 配对数不是框数。N=32 时图内部枚举出 128 个候选配对（不是 32），
    // 所以 host 缓冲必须按引擎的真实输出维度分配。踩过的坑：按框数分配时
    // sub_idx/obj_idx 读越界，拿到 -1063895040 这类垃圾值，解码于是全部
    // 被 in_range 过滤掉，表现为"推理成功但 0 条关系"。
    pairs_ = 0;
    for (const auto& t : core_.tensors()) {
        if (t.name != kOutPred || t.nb_dims < 2)
            continue;
        if (t.d[1] > 0)
            pairs_ = static_cast<size_t>(t.d[1]);
    }
    if (pairs_ == 0) {
        LOG_ERROR("[Relation] cannot read num_pairs from engine's pred_logits dims");
        unload();
        return false;
    }

    if (!buildPredicateInputs(err)) {
        LOG_ERROR_FMT("[Relation] predicate input build failed: {}", err);
        unload();
        return false;
    }

    h_pred_.assign(pairs_ * static_cast<size_t>(fixed_predicates_), 0.0f);
    h_pair_.assign(pairs_, 0.0f);
    h_sub_.assign(pairs_, 0);
    h_obj_.assign(pairs_, 0);
    h_valid_.assign(pairs_, 0);
    LOG_INFO_FMT("[Relation] engine dims: boxes={} pairs={} predicates={}",
                 fixed_boxes_, pairs_, fixed_predicates_);

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
    // dim 从 W/alpha 的长度比推出（不是 alpha.size()，那是谓词个数不是维度）
    const size_t dim = bank_.dim();
    if (dim == 0 || bank_.embed_W.size() != dim * bank_.names.size()) {
        err = "W size mismatch: expect names * (W.size/alpha.size)";
        return false;
    }
    // 与 manifest 声明的 text_dim 交叉校验。bank 导错了行数时这是唯一能兜住的地方。
    if (bank_.text_dim > 0 && static_cast<size_t>(bank_.text_dim) != dim) {
        err = "text_dim=" + std::to_string(bank_.text_dim) +
              " disagrees with W/alpha implied dim=" + std::to_string(dim);
        return false;
    }
    return true;
}

bool TensorrtRelation::buildPredicateInputs(std::string& err) {
    if (bank_.empty()) {
        // 引擎内置谓词：W/alpha 是可选输入，不绑定
        return true;
    }
    // 图按**整张谓词表**导出（num_predicates 是静态维），所以 W/alpha 必须
    // 填满整表；只把激活谓词的行拷进去，其余填 0（那些谓词得分恒为基准，
    // 不会挤进 top-k）。若按激活数填，形状对不上，TRT 会直接拒绝。
    const size_t dim = bank_.dim();
    const size_t vocab = bank_.names.size();
    h_w_.assign(vocab * dim, 0.0f);
    h_alpha_.assign(vocab, 0.0f);

    for (const auto& name : active_predicates_) {
        const auto it = std::find(bank_.names.begin(), bank_.names.end(), name);
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
    fixed_predicates_ = bank_.empty()
                            ? static_cast<int64_t>(active_predicates_.size())
                            : static_cast<int64_t>(bank_.names.size());

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
    // 必须用 allocBuffer 而不是 buffer()：前者会 cudaMalloc 并把地址绑到
    // execution context，buffer() 只是查表。漏掉这步 enqueue 时拿到的是空地址，
    // 表现为 "uploadImage failed" 而没有任何 TRT 报错。
    void* dst = core_.allocBuffer(kInImage, nchw.size() * sizeof(float));
    if (dst == nullptr) {
        LOG_ERROR_FMT("[Relation] allocBuffer({}) failed", kInImage);
        return false;
    }
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

void TensorrtRelation::allocOutput(const char* name, size_t bytes) {
    if (core_.allocBuffer(name, bytes) == nullptr)
        LOG_ERROR_FMT("[Relation] allocBuffer({}) failed", name);
}

void TensorrtRelation::bindOutputAddresses() {
    // TRT 要求 enqueue 前所有 I/O 张量都已 setTensorAddress。这里只分配+绑定，
    // 不做 D2H——真正取数在 enqueue 之后的 copyOutputsToHost()。
    // 顺序写反的表现是 enqueueV3 报 "Neither address nor allocator is set for
    // output tensor pred_logits"，而不是任何形状或显存错误。
    allocOutput(kOutPred, h_pred_.size() * sizeof(float));
    allocOutput(kOutPair, h_pair_.size() * sizeof(float));
    // 元素类型必须与 ONNX 一致（见 relateanything.onnx 的 graph.output）：
    //   sub_idx / obj_idx = int64、valid_mask = bool(1 byte)。
    // 按 int32 读 int64 会错位，按 float 读 int64 更是拿到垃圾——两者都不报错，
    // 只会让 valid 全 0、sub/obj 指向越界，最后表现为"0 条关系"。
    allocOutput(kOutSub, h_sub_.size() * sizeof(int64_t));
    allocOutput(kOutObj, h_obj_.size() * sizeof(int64_t));
    allocOutput(kOutValid, h_valid_.size() * sizeof(uint8_t));
}

void TensorrtRelation::copyOutputsToHost() {
    // 地址已在 bindOutputAddresses() 里分配并绑定，这里只取数。
    // 若这里返回空就说明绑定漏了，必须报错而不是静默跳过——
    // 静默跳过的后果是解码拿到全零，看起来"推理成功但没有关系"。
    auto pull = [&](const char* name, void* host, size_t bytes) {
        void* dst = core_.buffer(name);
        if (dst == nullptr) {
            LOG_ERROR_FMT("[Relation] output {} was never bound; "
                          "bindOutputAddresses() is missing this tensor", name);
            return;
        }
        cudaError_t e = cudaMemcpyAsync(host, dst, bytes, cudaMemcpyDeviceToHost,
                                        static_cast<cudaStream_t>(core_.stream()));
        if (e != cudaSuccess)
            LOG_ERROR_FMT("[Relation] D2H {} failed: {}", name, cudaGetErrorString(e));
    };
    pull(kOutPred, h_pred_.data(), h_pred_.size() * sizeof(float));
    pull(kOutPair, h_pair_.data(), h_pair_.size() * sizeof(float));
    pull(kOutSub, h_sub_.data(), h_sub_.size() * sizeof(int64_t));
    pull(kOutObj, h_obj_.data(), h_obj_.size() * sizeof(int64_t));
    pull(kOutValid, h_valid_.data(), h_valid_.size() * sizeof(uint8_t));
}

bool TensorrtRelation::infer(const RelationInput& input,
                             std::vector<RelationTriplet>& out) {
    out.clear();
    if (!uploadImage(input)) {
        LOG_ERROR_FMT("[Relation] uploadImage failed");
        return false;
    }
    if (!runBoxes(input, out)) {
        LOG_ERROR_FMT("[Relation] runBoxes failed");
        return false;
    }
    return true;
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
        // 维度必须与 ONNX 声明一致，否则 setInputShape 会被 TRT 拒绝
        // （trtexec 建的 profile 是 box_counts:1 / alpha:243，都是 1-D）。
        // 实测踩过的坑：把 box_counts 设成 [1,1] 会直接
        // "profile 0 has 2 dimensions"，推理阶段同样失败。
        int64_t d[4] = {1, 3, bank_.img_size, bank_.img_size};
        if (!core_.setInputShape(kInImage, d, 4)) {
            LOG_ERROR_FMT("[Relation] setInputShape({}) failed", kInImage);
            return false;
        }
        int64_t b[3] = {1, k, 4};
        if (!core_.setInputShape(kInBoxes, b, 3)) {
            LOG_ERROR_FMT("[Relation] setInputShape({}) failed", kInBoxes);
            return false;
        }
        int64_t c[1] = {1};                       // box_counts 是 [batch]，1-D
        if (!core_.setInputShape(kInCounts, c, 1)) {
            LOG_ERROR_FMT("[Relation] setInputShape({}) failed", kInCounts);
            return false;
        }
        // W / alpha 的第一维是**静态**的整张谓词表（引擎按 243 导出）。
        // 想只激活几个谓词，就在表里把对应行填好、未激活行填 0，
        // 靠 host 侧阈值过滤——不能按激活数去 setInputShape，
        // TRT 会直接报 "Static dimension mismatch"。
        if (!h_w_.empty()) {
            const int64_t vp = static_cast<int64_t>(bank_.names.size());
            int64_t w[2] = {vp, static_cast<int64_t>(bank_.dim())};
            if (!core_.setInputShape(kInW, w, 2)) {
                LOG_ERROR_FMT("[Relation] setInputShape({}) failed", kInW);
                return false;
            }
            int64_t a[1] = {vp};                  // alpha 与 W 同为整表大小
            if (!core_.setInputShape(kInAlpha, a, 1)) {
                LOG_ERROR_FMT("[Relation] setInputShape({}) failed", kInAlpha);
                return false;
            }
        }
    }

    // 框：必须转成**归一化 cxcywh**（0..1），不是 xyxy 像素坐标。
    //
    // 依据 deploy/export_onnx.py:159 —— `boxes = torch.rand(1,N,4)*0.5+0.25
    // # cxcywh in [0,1]`，以及 deploy/runtime.py:412-414 的
    //   b[:,[0,2]] /= W; b[:,[1,3]] /= H; boxes = [cx,cy,w,h]
    // 传 xyxy 像素坐标不会报错，只会算出无意义的关系分（实测 0 条输出）。
    // 注意除的是**图像边长**（letterbox 后为 img_size），因为图像本身已缩放到
    // img_size，所以像素坐标除以 img_size 即得归一化值。
    const float side = static_cast<float>(bank_.img_size);
    std::vector<float> boxes(static_cast<size_t>(k) * 4, 0.0f);
    box_scores_.assign(static_cast<size_t>(n_used), 0.0f);
    for (int64_t i = 0; i < n_used; ++i) {
        const auto& b = input.boxes[static_cast<size_t>(i)];
        const float x1 = b.x1 / side, y1 = b.y1 / side;
        const float x2 = b.x2 / side, y2 = b.y2 / side;
        boxes[static_cast<size_t>(i) * 4 + 0] = (x1 + x2) / 2.0f;  // cx
        boxes[static_cast<size_t>(i) * 4 + 1] = (y1 + y2) / 2.0f;  // cy
        boxes[static_cast<size_t>(i) * 4 + 2] = (x2 - x1);         // w
        boxes[static_cast<size_t>(i) * 4 + 3] = (y2 - y1);         // h
        box_scores_[static_cast<size_t>(i)] = b.confidence;
    }

    void* d_boxes = core_.allocBuffer(kInBoxes, boxes.size() * sizeof(float));
    void* d_counts = core_.allocBuffer(kInCounts, sizeof(int32_t));
    if (d_boxes == nullptr || d_counts == nullptr) {
        LOG_ERROR("[Relation] allocBuffer for boxes/box_counts failed");
        return false;
    }
    cudaMemcpyAsync(d_boxes, boxes.data(), boxes.size() * sizeof(float),
                    cudaMemcpyHostToDevice, static_cast<cudaStream_t>(core_.stream()));
    const int32_t count32 = static_cast<int32_t>(n_used);
    cudaMemcpyAsync(d_counts, &count32, sizeof(int32_t),
                    cudaMemcpyHostToDevice, static_cast<cudaStream_t>(core_.stream()));
    if (!h_w_.empty()) {
        void* d_w = core_.allocBuffer(kInW, h_w_.size() * sizeof(float));
        void* d_a = core_.allocBuffer(kInAlpha, h_alpha_.size() * sizeof(float));
        if (d_w) cudaMemcpyAsync(d_w, h_w_.data(), h_w_.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, static_cast<cudaStream_t>(core_.stream()));
        if (d_a) cudaMemcpyAsync(d_a, h_alpha_.data(), h_alpha_.size() * sizeof(float),
                                 cudaMemcpyHostToDevice, static_cast<cudaStream_t>(core_.stream()));
    }

    bindOutputAddresses();
    if (!runGraph()) {
        LOG_WARN("[Relation] graph execution failed");
        return false;
    }
    copyOutputsToHost();
    core_.synchronize();

    // 每帧一次汇总。valid 为 0 是"这帧没有可用配对"的强信号：通常意味着
    // 框坐标格式错（图要归一化 cxcywh）或 box_counts 与实际框数不一致。
    {
        size_t valid_cnt = 0;
        for (uint8_t v : h_valid_) valid_cnt += (v != 0) ? 1u : 0u;
        LOG_DEBUG_FMT("[Relation] pairs={} valid={} boxes_used={}",
                      h_valid_.size(), valid_cnt, n_used);
    }

    // pred_logits 是 [pairs, 整表]，直接整段传给解码器（列名给整表，
    // 激活过滤由 only_predicates 负责）。
    std::vector<float> pred(h_pred_.begin(), h_pred_.end());

    std::vector<DecodedRelation> decoded;
    decodeRelations(pred, h_pair_, h_sub_, h_obj_, h_valid_, bank_.names,
                    thresholds_, box_scores_, decoded, active_predicates_);
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
