// src/hal/infer/rk/rknn_pose_estimation.cpp
// RKNN 姿态估计推理引擎实现——与 rknn_detection_engine 相同的 dlopen 桩模式
#include "rknn_pose_estimation.h"
#include "ai_stream/hal/pose_estimation_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <algorithm>
#include <cstring>
#include <fstream>

namespace ai_stream {
namespace hal {

namespace {

// ---- dlopen 桩（与 rknn_detection_engine 相同模式，静态符号不冲突）----
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
        LOG_WARN_FMT("[RknnPoseEstimation] dlopen librknnrt.so failed: {}", dlerror());
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
    if (!ok) LOG_ERROR("[RknnPoseEstimation] Failed to load RKNN API symbols");
    return ok;
}

// 压缩 dims（去掉 1 值维度）
std::vector<int> squeezeDims(const rknn_tensor_attr& attr) {
    std::vector<int> dims;
    for (uint32_t i = 0; i < attr.n_dims; ++i) {
        if (attr.dims[i] > 1) dims.push_back(attr.dims[i]);
    }
    return dims;
}

constexpr int POSE_DIM = 56;  // 4 box + 1 score + 51 kpt

} // namespace

RknnPoseEstimation::RknnPoseEstimation() = default;

RknnPoseEstimation::~RknnPoseEstimation() {
    if (rknn_ctx_ && p_destroy) {
        p_destroy(rknn_ctx_);
    }
    rknn_ctx_ = 0;
}

bool RknnPoseEstimation::isAvailable() const {
    return load_rknn_lib();
}

size_t RknnPoseEstimation::getOutputFloatsPerPerson() const {
    return static_cast<size_t>(num_anchors_) * POSE_DIM;
}

std::pair<int, int> RknnPoseEstimation::getInputSize() const {
    return {input_width_, input_height_};
}

bool RknnPoseEstimation::loadModel(const PoseEstimationConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    input_width_ = config.input_width;
    input_height_ = config.input_height;

    if (!load_rknn_lib()) return false;

    std::ifstream fin(config.model_path, std::ios::binary);
    if (!fin.is_open()) {
        LOG_ERROR_FMT("[RknnPoseEstimation] Model file not found: {}", config.model_path);
        return false;
    }
    fin.seekg(0, std::ios::end);
    const size_t model_size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(model_size);
    if (!fin.read(reinterpret_cast<char*>(model_data.data()), model_size)) {
        LOG_ERROR("[RknnPoseEstimation] Failed to read model data");
        return false;
    }
    fin.close();

    int ret = p_init(&rknn_ctx_, model_data.data(),
                     static_cast<uint32_t>(model_size), 0, nullptr);
    if (ret != RKNN_SUCC) {
        LOG_ERROR_FMT("[RknnPoseEstimation] rknn_init failed: {}", ret);
        rknn_ctx_ = 0;
        return false;
    }
    p_set_core_mask(rknn_ctx_, RKNN_NPU_CORE_AUTO);

    // 输入属性
    rknn_tensor_attr in_attr;
    memset(&in_attr, 0, sizeof(in_attr));
    in_attr.index = 0;
    if (p_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &in_attr, sizeof(in_attr)) != RKNN_SUCC) {
        LOG_ERROR("[RknnPoseEstimation] query input attr failed");
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

    // 输出属性
    rknn_input_output_num io_num{};
    memset(&io_num, 0, sizeof(io_num));
    if (p_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) != RKNN_SUCC) {
        LOG_ERROR("[RknnPoseEstimation] query in/out num failed");
        return false;
    }
    if (io_num.n_output != 1) {
        LOG_ERROR_FMT("[RknnPoseEstimation] expected single output, got {}", io_num.n_output);
        return false;
    }
    n_outputs_ = io_num.n_output;

    rknn_tensor_attr out_attr;
    memset(&out_attr, 0, sizeof(out_attr));
    out_attr.index = 0;
    if (p_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &out_attr, sizeof(out_attr)) != RKNN_SUCC) {
        LOG_ERROR("[RknnPoseEstimation] query output attr failed");
        return false;
    }

    const std::vector<int> d = squeezeDims(out_attr);
    if (d.size() == 2) {
        if (d[0] == POSE_DIM && d[1] > POSE_DIM) {
            out_row_major_ = false;   // [56, N] 通道优先，需转置
            num_anchors_ = d[1];
        } else if (d[1] == POSE_DIM && d[0] > POSE_DIM) {
            out_row_major_ = true;    // [N, 56] 行主序，直接拷贝
            num_anchors_ = d[0];
        } else {
            LOG_ERROR_FMT("[RknnPoseEstimation] unsupported output dims: {}x{}", d[0], d[1]);
            return false;
        }
    } else if (d.size() == 3 && d[0] == POSE_DIM) {
        // [56, h, w]（多尺度分支未 concat 的导出形态），anchors = h*w
        out_row_major_ = false;
        num_anchors_ = d[1] * d[2];
    } else {
        LOG_ERROR("[RknnPoseEstimation] unsupported output rank");
        return false;
    }

    // 节点 decodeFrame 固定 NUM_CANDIDATES=8400 / POSE_DIM=56，强制契约一致
    if (num_anchors_ != 8400) {
        LOG_ERROR_FMT("[RknnPoseEstimation] anchors={} != 8400 (node fixed candidate contract)",
                      num_anchors_);
        return false;
    }

    loaded_ = true;
    LOG_INFO_FMT("[RknnPoseEstimation] Model loaded: {} (input {}x{} type={} fmt={}, "
                 "anchors={}, layout={})",
                 config.model_path, input_width_, input_height_,
                 static_cast<int>(input_type_), static_cast<int>(input_fmt_),
                 num_anchors_, out_row_major_ ? "row-major [N,56]" : "channels-first [56,N]");
    return true;
}

bool RknnPoseEstimation::inferHost(const float* input_nchw, int num_persons,
                                   std::vector<float>& output_host) {
    if (!loaded_) {
        LOG_ERROR("[RknnPoseEstimation] Model not loaded");
        return false;
    }
    if (!input_nchw || num_persons <= 0) return false;

    std::lock_guard<std::mutex> lock(mutex_);

    output_host.resize(static_cast<size_t>(num_persons) * num_anchors_ * POSE_DIM);

    const int hw = input_width_ * input_height_;
    const int person_stride = 3 * hw;

    for (int p = 0; p < num_persons; ++p) {
        float* out_person = output_host.data() + static_cast<size_t>(p) * num_anchors_ * POSE_DIM;
        if (!inferOne(input_nchw + static_cast<size_t>(p) * person_stride, out_person)) {
            LOG_ERROR_FMT("[RknnPoseEstimation] inference failed for person {}", p);
            return false;
        }
    }
    return true;
}

bool RknnPoseEstimation::inferOne(const float* input_nchw, float* out_person) {
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
                float v = planes[c][idx];
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
        LOG_ERROR("[RknnPoseEstimation] rknn_inputs_set failed");
        return false;
    }
    if (p_run(rknn_ctx_, nullptr) != RKNN_SUCC) {
        LOG_ERROR("[RknnPoseEstimation] rknn_run failed");
        return false;
    }

    // 2. 取输出（want_float=1 统一 float）
    rknn_output out{};
    out.index = 0;
    out.want_float = 1;
    if (p_outputs_get(rknn_ctx_, 1, &out, nullptr) != RKNN_SUCC) {
        LOG_ERROR("[RknnPoseEstimation] rknn_outputs_get failed");
        return false;
    }

    const float* src = static_cast<const float*>(out.buf);
    const size_t n_floats = out.size / sizeof(float);
    if (n_floats < static_cast<size_t>(num_anchors_) * POSE_DIM) {
        LOG_ERROR_FMT("[RknnPoseEstimation] output size mismatch: {} < {}",
                      n_floats, static_cast<size_t>(num_anchors_) * POSE_DIM);
        p_outputs_release(rknn_ctx_, 1, &out);
        return false;
    }

    // 3. 布局归一：统一为行主序 [N, 56]
    if (out_row_major_) {
        std::memcpy(out_person, src, static_cast<size_t>(num_anchors_) * POSE_DIM * sizeof(float));
    } else {
        // 通道优先 [56, N] → 转置为 [N, 56]
        for (int i = 0; i < num_anchors_; ++i) {
            float* dst = out_person + static_cast<size_t>(i) * POSE_DIM;
            for (int c = 0; c < POSE_DIM; ++c) {
                dst[c] = src[static_cast<size_t>(c) * num_anchors_ + i];
            }
        }
    }

    p_outputs_release(rknn_ctx_, 1, &out);
    return true;
}

// 注册到工厂（RKNN 后端仅在有 NPU 的目标机可用，dlopen 失败则 isAvailable 为 false）
#ifdef WITH_RKNN
REGISTER_POSE_ESTIMATION_BACKEND(PoseEstimationBackend::RKNN, RknnPoseEstimation)
#endif

} // namespace hal
} // namespace ai_stream