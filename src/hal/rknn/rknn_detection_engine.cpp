// src/hal/rknn/rknn_detection_engine.cpp
// RKNN 检测推理引擎实现——见头文件说明
#include "rknn_detection_engine.h"
#include "ai_stream/hal/detection_inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace ai_stream {
namespace hal {

namespace {

// ---- dlopen 桩（与 rknn_inference_engine 相同模式，静态符号不冲突）----
using rknn_init_fn = int (*)(rknn_context*, void*, uint32_t, uint32_t, void*);
using rknn_destroy_fn = int (*)(rknn_context);
using rknn_query_fn = int (*)(rknn_context, int, void*, uint32_t);
using rknn_inputs_set_fn = int (*)(rknn_context, uint32_t, rknn_input*);
using rknn_run_fn = int (*)(rknn_context, void*);
using rknn_outputs_get_fn = int (*)(rknn_context, uint32_t, rknn_output*, void*);
using rknn_outputs_release_fn = int (*)(rknn_context, uint32_t, rknn_output*);
using rknn_set_core_mask_fn = int (*)(rknn_context, int);

rknn_init_fn p_init = nullptr;
rknn_destroy_fn p_destroy = nullptr;
rknn_query_fn p_query = nullptr;
rknn_inputs_set_fn p_inputs_set = nullptr;
rknn_run_fn p_run = nullptr;
rknn_outputs_get_fn p_outputs_get = nullptr;
rknn_outputs_release_fn p_outputs_release = nullptr;
rknn_set_core_mask_fn p_set_core_mask = nullptr;

bool load_rknn_lib() {
    static bool tried = false;
    static bool ok = false;
    if (tried) return ok;
    tried = true;
    void* h = dlopen("librknnrt.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        LOG_WARN_FMT("[RknnDetectionEngine] dlopen librknnrt.so failed: {}", dlerror());
        return false;
    }
    auto sym = [&](const char* n) { return dlsym(h, n); };
    p_init = reinterpret_cast<rknn_init_fn>(sym("rknn_init"));
    p_destroy = reinterpret_cast<rknn_destroy_fn>(sym("rknn_destroy"));
    p_query = reinterpret_cast<rknn_query_fn>(sym("rknn_query"));
    p_inputs_set = reinterpret_cast<rknn_inputs_set_fn>(sym("rknn_inputs_set"));
    p_run = reinterpret_cast<rknn_run_fn>(sym("rknn_run"));
    p_outputs_get = reinterpret_cast<rknn_outputs_get_fn>(sym("rknn_outputs_get"));
    p_outputs_release = reinterpret_cast<rknn_outputs_release_fn>(sym("rknn_outputs_release"));
    p_set_core_mask = reinterpret_cast<rknn_set_core_mask_fn>(sym("rknn_set_core_mask"));
    ok = p_init && p_destroy && p_query && p_inputs_set && p_run &&
         p_outputs_get && p_outputs_release && p_set_core_mask;
    if (!ok) LOG_ERROR("[RknnDetectionEngine] Failed to load RKNN API symbols");
    return ok;
}

// DFL：16 bins softmax 加权求和 → 该边距离
inline float dfl_sum(const float* bins /*16*/) {
    float maxv = bins[0];
    for (int i = 1; i < 16; ++i) maxv = std::max(maxv, bins[i]);
    float e[16], sum = 0.0f;
    for (int i = 0; i < 16; ++i) {
        e[i] = std::exp(bins[i] - maxv);
        sum += e[i];
    }
    float acc = 0.0f;
    for (int i = 0; i < 16; ++i) acc += (i * e[i]) / sum;
    return acc;
}

inline float iou_xyxy(const float a[4], const float b[4]) {
    const float x1 = std::max(a[0], b[0]);
    const float y1 = std::max(a[1], b[1]);
    const float x2 = std::min(a[2], b[2]);
    const float y2 = std::min(a[3], b[3]);
    const float inter = std::max(0.0f, x2 - x1) * std::max(0.0f, y2 - y1);
    const float uni = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter;
    return uni > 0.0f ? inter / uni : 0.0f;
}

// 压缩 dims（去掉 1 值维度）
std::vector<int> squeezeDims(const rknn_tensor_attr& attr) {
    std::vector<int> dims;
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        if (attr.dims[i] > 1) dims.push_back(attr.dims[i]);
    }
    return dims;
}

} // namespace

RknnDetectionEngine::RknnDetectionEngine() = default;

RknnDetectionEngine::~RknnDetectionEngine() {
    if (rknn_ctx_ && p_destroy) {
        p_destroy(rknn_ctx_);
    }
    rknn_ctx_ = 0;
}

bool RknnDetectionEngine::isAvailable() const {
    return load_rknn_lib();
}

bool RknnDetectionEngine::loadModel(const DetectionInferenceConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    input_width_ = config.input_width;
    input_height_ = config.input_height;
    max_detections_ = config.max_detections > 0 ? config.max_detections : 200;
    // 归一化参数由预处理节点完成（resize_normalize），引擎不做二次归一化；
    // mean_/std_ 仅用于 UINT8 输入路径的反归一化，默认恒等。
    mean_ = {0.0f, 0.0f, 0.0f};
    std_ = {1.0f, 1.0f, 1.0f};

    if (!load_rknn_lib()) return false;

    std::ifstream fin(config.model_path, std::ios::binary);
    if (!fin.is_open()) {
        LOG_ERROR_FMT("[RknnDetectionEngine] Model file not found: {}", config.model_path);
        return false;
    }
    fin.seekg(0, std::ios::end);
    const size_t model_size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(model_size);
    if (!fin.read(reinterpret_cast<char*>(model_data.data()), model_size)) {
        LOG_ERROR("[RknnDetectionEngine] Failed to read model data");
        return false;
    }
    fin.close();

    int ret = p_init(&rknn_ctx_, model_data.data(),
                     static_cast<uint32_t>(model_size), 0, nullptr);
    if (ret != RKNN_SUCC) {
        LOG_ERROR_FMT("[RknnDetectionEngine] rknn_init failed: {}", ret);
        rknn_ctx_ = 0;
        return false;
    }
    p_set_core_mask(rknn_ctx_, RKNN_NPU_CORE_AUTO);

    // 输入属性
    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    if (p_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)) != RKNN_SUCC) {
        LOG_ERROR("[RknnDetectionEngine] query input attr failed");
        return false;
    }
    input_type_ = in_attr.type;
    input_fmt_ = in_attr.fmt;
    if (in_attr.n_dims >= 4) {
        if (in_attr.fmt == RKNN_TENSOR_NHWC) {
            input_height_ = in_attr.dims[1] > 0 ? in_attr.dims[1] : input_height_;
            input_width_ = in_attr.dims[2] > 0 ? in_attr.dims[2] : input_width_;
        } else {
            input_height_ = in_attr.dims[2] > 0 ? in_attr.dims[2] : input_height_;
            input_width_ = in_attr.dims[3] > 0 ? in_attr.dims[3] : input_width_;
        }
    }
    input_need_denorm_ =
        (in_attr.type == RKNN_TENSOR_UINT8 || in_attr.type == RKNN_TENSOR_INT8);

    if (!detectOutputLayout()) return false;

    loaded_ = true;
    LOG_INFO_FMT("[RknnDetectionEngine] Model loaded: {} (input {}x{} type={} fmt={}, "
                 "layout={}, branches={}, classes={})",
                 config.model_path, input_width_, input_height_,
                 static_cast<int>(input_type_), static_cast<int>(input_fmt_),
                 static_cast<int>(layout_), branches_.size(), num_classes_);
    return allocateOutputBuffers();
}

bool RknnDetectionEngine::detectOutputLayout() {
    rknn_input_output_num io_num{};
    memset(&io_num, 0, sizeof(io_num));
    if (p_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != RKNN_SUCC) {
        LOG_ERROR("[RknnDetectionEngine] query in/out num failed");
        return false;
    }
    n_inputs_ = io_num.n_input;
    n_outputs_ = io_num.n_output;

    // 收集各输出 squeeze 后的 dims
    std::vector<std::vector<int>> out_dims(n_outputs_);
    for (uint32_t i = 0; i < n_outputs_; ++i) {
        rknn_tensor_attr attr;
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        if (p_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr)) != RKNN_SUCC) {
            LOG_ERROR_FMT("[RknnDetectionEngine] query output[{}] attr failed", i);
            return false;
        }
        out_dims[i] = squeezeDims(attr);
    }

    // ---- 单输出：V8_TRANS / V8_DFL(concat) / V5 ----
    if (n_outputs_ == 1) {
        const auto& d = out_dims[0];
        if (d.size() < 2) {
            LOG_ERROR("[RknnDetectionEngine] output dims too small");
            return false;
        }
        const int a = d[0];
        const int b = d[1];
        if (a < 64) {  // [1, 4+nc, N]
            num_classes_ = a - 4;
            num_anchors_ = b;
            layout_ = OutputLayout::V8_TRANS;
        } else if (a >= 68 && a <= 64 + 365) {  // [1, 64+nc, N] concat DFL
            num_classes_ = a - 64;
            num_anchors_ = b;
            layout_ = OutputLayout::V8_DFL;
            branches_.clear();  // 单 concat：按 8400 三尺度分段
        } else if (b < 64 && b >= 5 && a > 100) {  // [1, N, 5+nc]
            num_anchors_ = a;
            num_classes_ = b - 5;
            layout_ = OutputLayout::V5;
        } else if (b >= 68) {  // [1, N, 64+nc] 罕见转置 DFL
            num_anchors_ = a;
            num_classes_ = b - 64;
            layout_ = OutputLayout::UNKNOWN;  // 暂不支持，明确报错
        } else {
            layout_ = OutputLayout::UNKNOWN;
        }
    } else if (n_outputs_ == 3) {
        // 3 分支 DFL：[1, 64+nc, h, w]，各分支通道数一致
        for (uint32_t i = 0; i < 3; ++i) {
            const auto& d = out_dims[i];
            if (d.size() < 3) {
                LOG_ERROR_FMT("[RknnDetectionEngine] branch {} dims unsupported", i);
                return false;
            }
            const int c = d[0];
            const int fw = d[d.size() - 1];
            const int fh = d[d.size() - 2];
            if (i == 0) {
                if (c < 68) {
                    LOG_ERROR_FMT("[RknnDetectionEngine] branch channels {} not DFL", c);
                    return false;
                }
                num_classes_ = c - 64;
                layout_ = OutputLayout::V8_DFL;
            } else if (c != num_classes_ + 64) {
                LOG_ERROR("[RknnDetectionEngine] branch channel mismatch");
                return false;
            }
            branches_.push_back({fw, input_width_ / fw});
        }
        num_anchors_ = 0;
        for (const auto& br : branches_) num_anchors_ += br.feat_w * (input_height_ / br.stride);
    } else {
        LOG_ERROR_FMT("[RknnDetectionEngine] unsupported output count {}", n_outputs_);
        return false;
    }

    if (layout_ == OutputLayout::UNKNOWN || num_classes_ <= 0 ||
        (n_outputs_ == 1 && num_anchors_ <= 0)) {
        LOG_ERROR_FMT("[RknnDetectionEngine] unsupported model output layout "
                      "(n_outputs={}, dims0={} x dims1={})",
                      n_outputs_,
                      out_dims[0].size() > 0 ? out_dims[0][0] : 0,
                      out_dims[0].size() > 1 ? out_dims[0][1] : 0);
        return false;
    }
    return true;
}

bool RknnDetectionEngine::allocateOutputBuffers() {
    const size_t cap = static_cast<size_t>(config_.max_batch_size) * max_detections_;
    out_boxes_.assign(cap * 4, 0.0f);
    out_scores_.assign(cap, 0.0f);
    out_classes_.assign(cap, 0);
    out_batch_ids_.assign(cap, 0);
    out_num_dets_ = 0;
    LOG_INFO_FMT("[RknnDetectionEngine] Output buffers allocated (cap={})", cap);
    return true;
}

bool RknnDetectionEngine::setInputTensor(const std::string& name, void* ptr) {
    tensor_ptrs_[name] = ptr;
    return true;
}

bool RknnDetectionEngine::setOutputTensor(const std::string& name, void* ptr) {
    tensor_ptrs_[name] = ptr;
    return true;
}

void* RknnDetectionEngine::getOutputTensor(const std::string& name) {
    if (name == boxes_name_) return out_boxes_.data();
    if (name == scores_name_) return out_scores_.data();
    if (name == classes_name_) return out_classes_.data();
    if (name == batch_ids_name_) return out_batch_ids_.data();
    if (name == num_dets_name_) return &out_num_dets_;
    auto it = tensor_ptrs_.find(name);
    return it != tensor_ptrs_.end() ? it->second : nullptr;
}

size_t RknnDetectionEngine::getOutputTensorSize(const std::string& name) const {
    if (name == boxes_name_) return out_boxes_.size() * sizeof(float);
    if (name == scores_name_) return out_scores_.size() * sizeof(float);
    if (name == classes_name_) return out_classes_.size() * sizeof(int64_t);
    if (name == batch_ids_name_) return out_batch_ids_.size() * sizeof(int64_t);
    if (name == num_dets_name_) return sizeof(int64_t);
    return 0;
}

bool RknnDetectionEngine::infer() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_) {
        LOG_ERROR("[RknnDetectionEngine] Model not loaded");
        return false;
    }
    auto it = tensor_ptrs_.find(input_name_);
    if (it == tensor_ptrs_.end() || !it->second) {
        LOG_ERROR("[RknnDetectionEngine] Input tensor not set");
        return false;
    }
    out_num_dets_ = 0;  // 每次推理重置累计
    return inferOne(static_cast<const float*>(it->second), 0);
}

bool RknnDetectionEngine::inferAsync(void* stream) {
    (void)stream;  // NPU 推理相对 CPU 天然异步，节点随后调用 synchronize
    return infer();
}

bool RknnDetectionEngine::synchronize(void* /*stream*/) {
    return true;
}

bool RknnDetectionEngine::inferOne(const float* input_nchw, int batch_slot) {
    // 1. 输入缓冲（UINT8 模型需反归一化 + NCHW→NHWC）
    void* in_buf = nullptr;
    uint32_t in_size = 0;
    rknn_input in{};
    in.index = 0;
    in.type = input_type_;
    in.fmt = input_fmt_;
    in.pass_through = 0;

    if (input_need_denorm_) {
        const int hw = input_width_ * input_height_;
        input_nhwc_.resize(static_cast<size_t>(hw) * 3);
        const float* planes[3] = {input_nchw, input_nchw + hw, input_nchw + 2 * hw};
        for (int idx = 0; idx < hw; ++idx) {
            uint8_t* dst = &input_nhwc_[static_cast<size_t>(idx) * 3];
            for (int c = 0; c < 3; ++c) {
                float v = planes[c][idx] * std_[c] + mean_[c];
                dst[c] = static_cast<uint8_t>(std::min(255.0f, std::max(0.0f, v)));
            }
        }
        in_buf = input_nhwc_.data();
        in_size = static_cast<uint32_t>(input_nhwc_.size());
    } else {
        in_buf = const_cast<float*>(input_nchw);
        in_size = static_cast<uint32_t>(input_width_) * input_height_ * 3 * sizeof(float);
    }
    in.buf = in_buf;
    in.size = in_size;

    if (p_inputs_set(rknn_ctx_, 1, &in) != RKNN_SUCC) {
        LOG_ERROR("[RknnDetectionEngine] rknn_inputs_set failed");
        return false;
    }
    if (p_run(rknn_ctx_, nullptr) != RKNN_SUCC) {
        LOG_ERROR("[RknnDetectionEngine] rknn_run failed");
        return false;
    }

    // 2. 取输出（want_float=1 统一 float）
    std::vector<rknn_output> outs(n_outputs_);
    for (uint32_t i = 0; i < n_outputs_; ++i) {
        outs[i].index = i;
        outs[i].want_float = 1;
    }
    if (p_outputs_get(rknn_ctx_, n_outputs_, outs.data(), nullptr) != RKNN_SUCC) {
        LOG_ERROR("[RknnDetectionEngine] rknn_outputs_get failed");
        return false;
    }

    std::vector<std::vector<float>> out_data(n_outputs_);
    for (uint32_t i = 0; i < n_outputs_; ++i) {
        const float* src = static_cast<const float*>(outs[i].buf);
        out_data[i].assign(src, src + outs[i].size / sizeof(float));
    }
    p_outputs_release(rknn_ctx_, n_outputs_, outs.data());

    // 3. 解码
    std::vector<float> cx, cy, bw, bh, score;
    std::vector<int> cls;
    if (!collectCandidates(out_data, cx, cy, bw, bh, score, cls)) return false;

    // 4. 按类别 NMS + 截断
    const int n = static_cast<int>(score.size());
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return score[a] > score[b]; });

    std::vector<int> keep;
    std::vector<bool> suppressed(n, false);
    for (int i = 0; i < n && static_cast<int>(keep.size()) < max_detections_; ++i) {
        const int idx = order[i];
        if (suppressed[idx]) continue;
        keep.push_back(idx);
        const float box_i[4] = {cx[idx] - bw[idx] / 2, cy[idx] - bh[idx] / 2,
                                cx[idx] + bw[idx] / 2, cy[idx] + bh[idx] / 2};
        for (int j = i + 1; j < n; ++j) {
            const int jdx = order[j];
            if (suppressed[jdx] || cls[jdx] != cls[idx]) continue;
            const float box_j[4] = {cx[jdx] - bw[jdx] / 2, cy[jdx] - bh[jdx] / 2,
                                    cx[jdx] + bw[jdx] / 2, cy[jdx] + bh[jdx] / 2};
            if (iou_xyxy(box_i, box_j) > nms_iou_) suppressed[jdx] = true;
        }
    }

    // 5. 写输出缓冲
    const int cap = static_cast<int>(out_scores_.size());
    const int kept = std::min(static_cast<int>(keep.size()), cap - static_cast<int>(out_num_dets_));
    for (int k = 0; k < kept; ++k) {
        const int idx = keep[k];
        const size_t o = out_num_dets_ + k;
        out_boxes_[o * 4 + 0] = cx[idx];
        out_boxes_[o * 4 + 1] = cy[idx];
        out_boxes_[o * 4 + 2] = bw[idx];
        out_boxes_[o * 4 + 3] = bh[idx];
        out_scores_[o] = score[idx];
        out_classes_[o] = cls[idx];
        out_batch_ids_[o] = batch_slot;
    }
    out_num_dets_ += kept;
    return true;
}

bool RknnDetectionEngine::collectCandidates(
    const std::vector<std::vector<float>>& outs,
    std::vector<float>& cx, std::vector<float>& cy,
    std::vector<float>& bw, std::vector<float>& bh,
    std::vector<float>& score, std::vector<int>& cls) {

    cx.clear(); cy.clear(); bw.clear(); bh.clear(); score.clear(); cls.clear();

    if (layout_ == OutputLayout::V8_TRANS) {
        // [1, 4+nc, N]：第 i 列为第 i 个 anchor
        const int N = num_anchors_;
        const float* data = outs[0].data();
        for (int i = 0; i < N; ++i) {
            float best = 0.0f;
            int best_c = -1;
            for (int c = 0; c < num_classes_; ++c) {
                const float v = data[(4 + c) * N + i];
                if (v > best) { best = v; best_c = c; }
            }
            if (best <= 0.0f || best_c < 0) continue;
            cx.push_back(data[0 * N + i]);
            cy.push_back(data[1 * N + i]);
            bw.push_back(data[2 * N + i]);
            bh.push_back(data[3 * N + i]);
            score.push_back(best);
            cls.push_back(best_c);
        }
        return true;
    }

    if (layout_ == OutputLayout::V5) {
        // [1, N, 5+nc]
        const int N = num_anchors_;
        const int stride = 5 + num_classes_;
        const float* data = outs[0].data();
        for (int i = 0; i < N; ++i) {
            const float* row = data + static_cast<size_t>(i) * stride;
            const float obj = row[4];
            if (obj <= 0.0f) continue;
            float best = 0.0f;
            int best_c = -1;
            for (int c = 0; c < num_classes_; ++c) {
                const float v = row[5 + c] * obj;
                if (v > best) { best = v; best_c = c; }
            }
            if (best <= 0.0f || best_c < 0) continue;
            cx.push_back(row[0]);
            cy.push_back(row[1]);
            bw.push_back(row[2]);
            bh.push_back(row[3]);
            score.push_back(best);
            cls.push_back(best_c);
        }
        return true;
    }

    // V8_DFL
    if (branches_.empty()) {
        // 单 concat [1, 64+nc, 8400]，三尺度分段：80²@8 / 40²@16 / 20²@32
        const float* data = outs[0].data();
        const int N = num_anchors_;
        const int base_fw = input_width_ / 8;
        for (int i = 0; i < N; ++i) {
            int fw, stride, anchor_in_branch;
            if (i < base_fw * base_fw) {
                fw = base_fw; stride = 8; anchor_in_branch = i;
            } else if (i < base_fw * base_fw + (base_fw / 2) * (base_fw / 2)) {
                fw = base_fw / 2; stride = 16;
                anchor_in_branch = i - base_fw * base_fw;
            } else {
                fw = base_fw / 4; stride = 32;
                anchor_in_branch = i - base_fw * base_fw - (base_fw / 2) * (base_fw / 2);
            }
            const int x = anchor_in_branch % fw;
            const int y = anchor_in_branch / fw;

            float best = 0.0f;
            int best_c = -1;
            for (int c = 0; c < num_classes_; ++c) {
                const float v = data[(64 + c) * N + i];
                if (v > best) { best = v; best_c = c; }
            }
            if (best <= 0.0f || best_c < 0) continue;

            float ltrb[4];
            for (int d = 0; d < 4; ++d) {
                float bins[16];
                for (int b = 0; b < 16; ++b) bins[b] = data[(d * 16 + b) * N + i];
                ltrb[d] = dfl_sum(bins);
            }
            const float px = x * static_cast<float>(stride);
            const float py = y * static_cast<float>(stride);
            cx.push_back(px + (ltrb[2] - ltrb[0]) / 2);
            cy.push_back(py + (ltrb[3] - ltrb[1]) / 2);
            bw.push_back(ltrb[2] - ltrb[0]);
            bh.push_back(ltrb[3] - ltrb[1]);
            score.push_back(best);
            cls.push_back(best_c);
        }
        return true;
    }

    // 多分支：逐分支 [1, 64+nc, h, w]（want_float 输出布局为 [1, 64+nc, h*w]）
    size_t offset = 0;  // 无效：各分支独立 buffer，直接遍历
    for (size_t bi = 0; bi < branches_.size(); ++bi) {
        const auto& br = branches_[bi];
        const int fw = br.feat_w;
        const int fh = input_height_ / br.stride;
        const int N = fw * fh;
        const float* data = outs[bi].data();
        for (int i = 0; i < N; ++i) {
            const int x = i % fw;
            const int y = i / fw;
            float best = 0.0f;
            int best_c = -1;
            for (int c = 0; c < num_classes_; ++c) {
                const float v = data[(64 + c) * N + i];
                if (v > best) { best = v; best_c = c; }
            }
            if (best <= 0.0f || best_c < 0) continue;

            float ltrb[4];
            for (int d = 0; d < 4; ++d) {
                float bins[16];
                for (int b = 0; b < 16; ++b) bins[b] = data[(d * 16 + b) * N + i];
                ltrb[d] = dfl_sum(bins);
            }
            const float px = x * static_cast<float>(br.stride);
            const float py = y * static_cast<float>(br.stride);
            cx.push_back(px + (ltrb[2] - ltrb[0]) / 2);
            cy.push_back(py + (ltrb[3] - ltrb[1]) / 2);
            bw.push_back(ltrb[2] - ltrb[0]);
            bh.push_back(ltrb[3] - ltrb[1]);
            score.push_back(best);
            cls.push_back(best_c);
        }
    }
    (void)offset;
    return true;
}

std::vector<std::string> RknnDetectionEngine::getInputNames() const {
    return {input_name_};
}

std::vector<std::string> RknnDetectionEngine::getOutputNames() const {
    return {boxes_name_, scores_name_, classes_name_, batch_ids_name_, num_dets_name_};
}

std::pair<int, int> RknnDetectionEngine::getInputSize() const {
    return {input_width_, input_height_};
}

int RknnDetectionEngine::getMaxBatchSize() const {
    return config_.max_batch_size;
}

// 注册到工厂
#ifdef WITH_RKNN
REGISTER_DETECTION_INFERENCE_BACKEND(DetectionBackend::RKNN, RknnDetectionEngine)
#endif

} // namespace hal
} // namespace ai_stream
