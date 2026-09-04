// src/hal/rknn/rknn_inference_engine.cpp
// RKNN 推理引擎后端——Rockchip RK3588 NPU
// 通过 dlopen/dlsym 动态加载 librknnrt.so，兼容 x86_64 编译主机
#include "rknn_inference_engine.h"
#include "ai_stream/hal/inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <dlfcn.h>
#include <fstream>
#include <cstring>
#include <unistd.h>

namespace ai_stream {
namespace hal {

// RKNN API 函数指针类型
typedef int (*rknn_init_fn)(rknn_context*, void*, uint32_t, uint32_t, void*);
typedef int (*rknn_destroy_fn)(rknn_context);
typedef int (*rknn_query_fn)(rknn_context, int, void*, uint32_t);
typedef int (*rknn_inputs_set_fn)(rknn_context, uint32_t, void*);
typedef int (*rknn_outputs_get_fn)(rknn_context, uint32_t, void*, void*);
typedef int (*rknn_outputs_release_fn)(rknn_context, uint32_t, void*);
typedef int (*rknn_run_fn)(rknn_context, void*);
typedef int (*rknn_set_core_mask_fn)(rknn_context, int);
typedef int (*rknn_set_batch_core_num_fn)(rknn_context, int);

static void* rknn_dl_handle = nullptr;
static rknn_init_fn p_rknn_init = nullptr;
static rknn_destroy_fn p_rknn_destroy = nullptr;
static rknn_query_fn p_rknn_query = nullptr;
static rknn_inputs_set_fn p_rknn_inputs_set = nullptr;
static rknn_outputs_get_fn p_rknn_outputs_get = nullptr;
static rknn_outputs_release_fn p_rknn_outputs_release = nullptr;
static rknn_run_fn p_rknn_run = nullptr;
static rknn_set_core_mask_fn p_rknn_set_core_mask = nullptr;
static rknn_set_batch_core_num_fn p_rknn_set_batch_core_num = nullptr;

static bool load_rknn_lib() {
    if (rknn_dl_handle) return true;
    rknn_dl_handle = dlopen("librknnrt.so", RTLD_NOW | RTLD_GLOBAL);
    if (!rknn_dl_handle) {
        LOG_WARN_FMT("[RknnInferenceEngine] dlopen librknnrt.so failed: {}", dlerror());
        return false;
    }
    p_rknn_init = (rknn_init_fn)dlsym(rknn_dl_handle, "rknn_init");
    p_rknn_destroy = (rknn_destroy_fn)dlsym(rknn_dl_handle, "rknn_destroy");
    p_rknn_query = (rknn_query_fn)dlsym(rknn_dl_handle, "rknn_query");
    p_rknn_inputs_set = (rknn_inputs_set_fn)dlsym(rknn_dl_handle, "rknn_inputs_set");
    p_rknn_outputs_get = (rknn_outputs_get_fn)dlsym(rknn_dl_handle, "rknn_outputs_get");
    p_rknn_outputs_release = (rknn_outputs_release_fn)dlsym(rknn_dl_handle, "rknn_outputs_release");
    p_rknn_run = (rknn_run_fn)dlsym(rknn_dl_handle, "rknn_run");
    p_rknn_set_core_mask = (rknn_set_core_mask_fn)dlsym(rknn_dl_handle, "rknn_set_core_mask");
    p_rknn_set_batch_core_num = (rknn_set_batch_core_num_fn)dlsym(rknn_dl_handle, "rknn_set_batch_core_num");
    if (!p_rknn_init || !p_rknn_destroy || !p_rknn_query || !p_rknn_inputs_set ||
        !p_rknn_outputs_get || !p_rknn_outputs_release || !p_rknn_run || !p_rknn_set_core_mask) {
        LOG_ERROR("[RknnInferenceEngine] Failed to load RKNN API symbols");
        dlclose(rknn_dl_handle);
        rknn_dl_handle = nullptr;
        return false;
    }
    return true;
}

static inline bool rknn_loaded() { return p_rknn_init != nullptr; }

RknnInferenceEngine::RknnInferenceEngine()
    : rknn_ctx_(0), ctx_flags_(0), core_mask_(0) {
    LOG_DEBUG("[RknnInferenceEngine] Constructor");
}

RknnInferenceEngine::~RknnInferenceEngine() {
    if (rknn_ctx_) {
        if (rknn_loaded()) p_rknn_destroy(rknn_ctx_);
        rknn_ctx_ = 0;
        LOG_DEBUG("[RknnInferenceEngine] Destructor");
    }
}

bool RknnInferenceEngine::loadModel(const InferenceConfig& config) {
    config_ = config;
    LOG_INFO_FMT("[RknnInferenceEngine] Loading model: {}", config.model_path);

    if (!load_rknn_lib()) {
        LOG_ERROR("[RknnInferenceEngine] librknnrt.so not available");
        return false;
    }

    // 1. 读取 .rknn 文件到内存
    std::ifstream fin(config.model_path, std::ios::binary);
    if (!fin.is_open()) {
        LOG_ERROR_FMT("[RknnInferenceEngine] Failed to open model: {}", config.model_path);
        return false;
    }
    fin.seekg(0, std::ios::end);
    size_t model_size = fin.tellg();
    fin.seekg(0, std::ios::beg);
    std::vector<uint8_t> model_data(model_size);
    if (!fin.read(reinterpret_cast<char*>(model_data.data()), model_size)) {
        LOG_ERROR("[RknnInferenceEngine] Failed to read model data");
        return false;
    }
    fin.close();

    // 2. rknn_init
    int ret = p_rknn_init(&rknn_ctx_, model_data.data(), static_cast<uint32_t>(model_size), 0, nullptr);
    if (ret != RKNN_SUCC) {
        LOG_ERROR_FMT("[RknnInferenceEngine] rknn_init failed: {}", ret);
        rknn_ctx_ = 0;
        return false;
    }
    LOG_INFO_FMT("[RknnInferenceEngine] rknn_init succeeded, ctx={}", static_cast<uint32_t>(rknn_ctx_));

    // 3. queryTensorAttr() 获取输入输出属性
    if (!queryTensorAttr()) {
        LOG_ERROR("[RknnInferenceEngine] queryTensorAttr failed");
        return false;
    }

    loaded_ = true;
    LOG_INFO_FMT("[RknnInferenceEngine] Model loaded: {} ({} bytes)", config.model_path, model_size);
    return true;
}

bool RknnInferenceEngine::infer(const void* input_data, size_t input_size,
                                 void* output_data, size_t output_size) {
    if (!loaded_ || !rknn_ctx_) {
        LOG_ERROR("[RknnInferenceEngine] Model not loaded");
        return false;
    }

    // 1. rknn_inputs_set
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = const_cast<void*>(input_data);
    input.size = static_cast<uint32_t>(input_size);
    input.pass_through = 0;
    input.type = RKNN_TENSOR_FLOAT32;
    input.fmt = RKNN_TENSOR_NCHW;

    int ret = p_rknn_inputs_set(rknn_ctx_, 1, &input);
    if (ret != RKNN_SUCC) {
        LOG_ERROR_FMT("[RknnInferenceEngine] rknn_inputs_set failed: {}", ret);
        return false;
    }

    // 2. rknn_run
    ret = p_rknn_run(rknn_ctx_, nullptr);
    if (ret != RKNN_SUCC) {
        LOG_ERROR_FMT("[RknnInferenceEngine] rknn_run failed: {}", ret);
        return false;
    }

    // 3. rknn_outputs_get
    rknn_output output;
    memset(&output, 0, sizeof(output));
    output.index = 0;
    output.want_float = 1;
    output.is_prealloc = 1;
    output.buf = output_data;
    output.size = static_cast<uint32_t>(output_size);

    rknn_output_extend extend;
    memset(&extend, 0, sizeof(extend));

    ret = p_rknn_outputs_get(rknn_ctx_, 1, &output, &extend);
    if (ret != RKNN_SUCC) {
        LOG_ERROR_FMT("[RknnInferenceEngine] rknn_outputs_get failed: {}", ret);
        return false;
    }

    // 4. rknn_outputs_release
    p_rknn_outputs_release(rknn_ctx_, 1, &output);

    return true;
}

std::pair<int, int> RknnInferenceEngine::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int RknnInferenceEngine::getBatchSize() const {
    return config_.batch_size;
}

bool RknnInferenceEngine::isAvailable() const {
#ifdef WITH_RKNN
    if (access("/dev/rknpu", F_OK) == 0) return true;
    if (rknn_ctx_) return true;
#endif
    return false;
}

void RknnInferenceEngine::setCoreMask(int core_mask) {
    core_mask_ = core_mask;
}

bool RknnInferenceEngine::initRknnContext() {
    return rknn_ctx_ != 0;
}

bool RknnInferenceEngine::queryTensorAttr() {
    if (!rknn_ctx_ || !rknn_loaded()) return false;

    rknn_input_output_num io_num;
    memset(&io_num, 0, sizeof(io_num));
    int ret = p_rknn_query(rknn_ctx_, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num));
    if (ret != RKNN_SUCC) return false;

    input_attrs_.resize(io_num.n_input);
    for (uint32_t i = 0; i < io_num.n_input; i++) {
        rknn_tensor_attr& attr = input_attrs_[i];
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = p_rknn_query(rknn_ctx_, RKNN_QUERY_INPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) return false;
    }

    output_attrs_.resize(io_num.n_output);
    for (uint32_t i = 0; i < io_num.n_output; i++) {
        rknn_tensor_attr& attr = output_attrs_[i];
        memset(&attr, 0, sizeof(attr));
        attr.index = i;
        ret = p_rknn_query(rknn_ctx_, RKNN_QUERY_OUTPUT_ATTR, &attr, sizeof(attr));
        if (ret != RKNN_SUCC) return false;
    }

    LOG_INFO_FMT("[RknnInferenceEngine] queryTensorAttr: {} inputs, {} outputs",
                 io_num.n_input, io_num.n_output);
    return true;
}

bool RknnInferenceEngine::setInputs(const void* input_data, size_t input_size) {
    if (!rknn_ctx_ || !rknn_loaded()) return false;
    rknn_input input;
    memset(&input, 0, sizeof(input));
    input.index = 0;
    input.buf = const_cast<void*>(input_data);
    input.size = static_cast<uint32_t>(input_size);
    input.pass_through = 0;
    input.type = RKNN_TENSOR_FLOAT32;
    input.fmt = RKNN_TENSOR_NCHW;
    int ret = p_rknn_inputs_set(rknn_ctx_, 1, &input);
    return ret == RKNN_SUCC;
}

bool RknnInferenceEngine::getOutputs(void* output_data, size_t output_size) {
    if (!rknn_ctx_ || !rknn_loaded()) return false;
    rknn_output output;
    memset(&output, 0, sizeof(output));
    output.index = 0;
    output.want_float = 1;
    output.is_prealloc = 1;
    output.buf = output_data;
    output.size = static_cast<uint32_t>(output_size);

    rknn_output_extend extend;
    memset(&extend, 0, sizeof(extend));
    int ret = p_rknn_outputs_get(rknn_ctx_, 1, &output, &extend);
    if (ret == RKNN_SUCC) {
        p_rknn_outputs_release(rknn_ctx_, 1, &output);
    }
    return ret == RKNN_SUCC;
}

// 注册 RKNN 后端到工厂
#ifdef WITH_RKNN
REGISTER_INFERENCE_BACKEND(InferenceBackend::RKNN, RknnInferenceEngine)
#endif

} // namespace hal
} // namespace ai_stream
