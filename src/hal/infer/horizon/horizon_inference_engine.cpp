// src/hal/infer/horizon/horizon_inference_engine.cpp
// Horizon BPU 推理引擎实现——地平线 RDK S100P
// 通过 dlopen/dlsym 动态加载 libdnn.so，兼容 x86_64 编译主机
#include "horizon_inference_engine.h"
#include "ai_stream/hal/inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <fstream>
#include <cstring>
#include <cstdlib>

// Horizon SDK headers (forward declarations for dlopen pattern)
extern "C" {
#include <hb_dnn.h>
#include <hb_ucp.h>
#include <hb_ucp_sys.h>
}

namespace ai_stream {
namespace hal {

// ---- dlopen 函数指针类型 ----
typedef const char* (*hbDNNGetVersion_fn)();
typedef int32_t (*hbDNNInitializeFromFiles_fn)(hbDNNPackedHandle_t*, char const**, int32_t);
typedef int32_t (*hbDNNRelease_fn)(hbDNNPackedHandle_t);
typedef int32_t (*hbDNNGetModelNameList_fn)(char const***, int32_t*, hbDNNPackedHandle_t);
typedef int32_t (*hbDNNGetModelHandle_fn)(hbDNNHandle_t*, hbDNNPackedHandle_t, char const*);
typedef int32_t (*hbDNNGetInputCount_fn)(int32_t*, hbDNNHandle_t);
typedef int32_t (*hbDNNGetOutputCount_fn)(int32_t*, hbDNNHandle_t);
typedef int32_t (*hbDNNGetInputTensorProperties_fn)(struct hbDNNTensorProperties*, hbDNNHandle_t, int32_t);
typedef int32_t (*hbDNNGetOutputTensorProperties_fn)(struct hbDNNTensorProperties*, hbDNNHandle_t, int32_t);
typedef int32_t (*hbDNNInferV2_fn)(hbUCPTaskHandle_t*, struct hbDNNTensor*, struct hbDNNTensor const*, hbDNNHandle_t);
typedef int32_t (*hbUCPWaitTaskDone_fn)(hbUCPTaskHandle_t, int32_t);
typedef int32_t (*hbUCPReleaseTask_fn)(hbUCPTaskHandle_t);
typedef int32_t (*hbUCPMem_fn)(struct hbUCPSysMem*, uint64_t, int32_t);
typedef int32_t (*hbUCPFree_fn)(struct hbUCPSysMem*);

static struct HorizonDnnApi {
    void* dl_handle = nullptr;
    hbDNNGetVersion_fn GetVersion = nullptr;
    hbDNNInitializeFromFiles_fn InitializeFromFiles = nullptr;
    hbDNNRelease_fn Release = nullptr;
    hbDNNGetModelNameList_fn GetModelNameList = nullptr;
    hbDNNGetModelHandle_fn GetModelHandle = nullptr;
    hbDNNGetInputCount_fn GetInputCount = nullptr;
    hbDNNGetOutputCount_fn GetOutputCount = nullptr;
    hbDNNGetInputTensorProperties_fn GetInputTensorProperties = nullptr;
    hbDNNGetOutputTensorProperties_fn GetOutputTensorProperties = nullptr;
    hbDNNInferV2_fn InferV2 = nullptr;
    hbUCPWaitTaskDone_fn WaitTaskDone = nullptr;
    hbUCPReleaseTask_fn ReleaseTask = nullptr;
    hbUCPMem_fn Malloc = nullptr;
    hbUCPFree_fn Free = nullptr;
    bool loaded = false;
} g_api;

static bool load_horizon_dnn_lib() {
    if (g_api.loaded) return true;

    g_api.dl_handle = dlopen("libdnn.so", RTLD_NOW | RTLD_GLOBAL);
    if (!g_api.dl_handle) {
        LOG_WARN_FMT("[HorizonInferenceEngine] dlopen libdnn.so failed: {}", dlerror());
        // fallback: try libhbrt4.so
        g_api.dl_handle = dlopen("libhbrt4.so", RTLD_NOW | RTLD_GLOBAL);
        if (!g_api.dl_handle) {
            LOG_WARN_FMT("[HorizonInferenceEngine] dlopen libhbrt4.so failed: {}", dlerror());
            return false;
        }
    }

    auto sym = [&](const char* name) -> void* {
        void* s = dlsym(g_api.dl_handle, name);
        if (!s) LOG_WARN_FMT("[HorizonInferenceEngine] dlsym {} failed: {}", name, dlerror());
        return s;
    };

    g_api.GetVersion = reinterpret_cast<hbDNNGetVersion_fn>(sym("hbDNNGetVersion"));
    g_api.InitializeFromFiles = reinterpret_cast<hbDNNInitializeFromFiles_fn>(sym("hbDNNInitializeFromFiles"));
    g_api.Release = reinterpret_cast<hbDNNRelease_fn>(sym("hbDNNRelease"));
    g_api.GetModelNameList = reinterpret_cast<hbDNNGetModelNameList_fn>(sym("hbDNNGetModelNameList"));
    g_api.GetModelHandle = reinterpret_cast<hbDNNGetModelHandle_fn>(sym("hbDNNGetModelHandle"));
    g_api.GetInputCount = reinterpret_cast<hbDNNGetInputCount_fn>(sym("hbDNNGetInputCount"));
    g_api.GetOutputCount = reinterpret_cast<hbDNNGetOutputCount_fn>(sym("hbDNNGetOutputCount"));
    g_api.GetInputTensorProperties = reinterpret_cast<hbDNNGetInputTensorProperties_fn>(sym("hbDNNGetInputTensorProperties"));
    g_api.GetOutputTensorProperties = reinterpret_cast<hbDNNGetOutputTensorProperties_fn>(sym("hbDNNGetOutputTensorProperties"));
    g_api.InferV2 = reinterpret_cast<hbDNNInferV2_fn>(sym("hbDNNInferV2"));
    g_api.WaitTaskDone = reinterpret_cast<hbUCPWaitTaskDone_fn>(sym("hbUCPWaitTaskDone"));
    g_api.ReleaseTask = reinterpret_cast<hbUCPReleaseTask_fn>(sym("hbUCPReleaseTask"));
    g_api.Malloc = reinterpret_cast<hbUCPMem_fn>(sym("hbUCPMalloc"));
    g_api.Free = reinterpret_cast<hbUCPFree_fn>(sym("hbUCPFree"));

    g_api.loaded = g_api.GetVersion && g_api.InitializeFromFiles && g_api.Release &&
                   g_api.GetModelNameList && g_api.GetModelHandle &&
                   g_api.GetInputCount && g_api.GetOutputCount &&
                   g_api.GetInputTensorProperties && g_api.GetOutputTensorProperties &&
                   g_api.InferV2 && g_api.WaitTaskDone && g_api.ReleaseTask;

    if (!g_api.loaded) {
        LOG_ERROR("[HorizonInferenceEngine] Failed to load required DNN API symbols");
        dlclose(g_api.dl_handle);
        g_api.dl_handle = nullptr;
        return false;
    }

    const char* ver = g_api.GetVersion();
    LOG_INFO_FMT("[HorizonInferenceEngine] Loaded Horizon DNN SDK: {}", ver ? ver : "unknown");
    return true;
}

HorizonInferenceEngine::HorizonInferenceEngine() {
    LOG_DEBUG("[HorizonInferenceEngine] Constructor");
}

HorizonInferenceEngine::~HorizonInferenceEngine() {
    if (dnn_packed_handle_) {
        if (g_api.loaded && g_api.Release) {
            g_api.Release(dnn_packed_handle_);
        }
        dnn_packed_handle_ = nullptr;
    }
    LOG_DEBUG("[HorizonInferenceEngine] Destructor");
}

bool HorizonInferenceEngine::loadModel(const InferenceConfig& config) {
    config_ = config;
    LOG_INFO_FMT("[HorizonInferenceEngine] Loading model: {}", config.model_path);

    if (!load_horizon_dnn_lib()) {
        LOG_ERROR("[HorizonInferenceEngine] Horizon DNN SDK not available");
        return false;
    }

    // Load model file
    std::string model_path = config.model_path;
    char const* model_files[] = {model_path.c_str()};
    int32_t ret = g_api.InitializeFromFiles(&dnn_packed_handle_, model_files, 1);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonInferenceEngine] Failed to initialize model: {} (error: {})",
                       model_path, ret);
        return false;
    }

    // Get model name and handle
    char const** model_name_list = nullptr;
    int32_t model_count = 0;
    ret = g_api.GetModelNameList(&model_name_list, &model_count, dnn_packed_handle_);
    if (ret != 0 || model_count <= 0) {
        LOG_ERROR("[HorizonInferenceEngine] Failed to get model name list");
        return false;
    }

    ret = g_api.GetModelHandle(&dnn_handle_, dnn_packed_handle_, model_name_list[0]);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonInferenceEngine] Failed to get model handle: {}", model_name_list[0]);
        return false;
    }

    if (!queryModelInfo()) {
        return false;
    }

    loaded_ = true;
    LOG_INFO_FMT("[HorizonInferenceEngine] Model loaded: {} ({}x{}, {} inputs, {} outputs)",
                 model_path, input_width_, input_height_, num_inputs_, num_outputs_);
    return true;
}

bool HorizonInferenceEngine::queryModelInfo() {
    if (!g_api.loaded || !dnn_handle_) return false;

    int32_t ret;

    // Query input count
    ret = g_api.GetInputCount(&num_inputs_, dnn_handle_);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonInferenceEngine] GetInputCount failed: {}", ret);
        return false;
    }

    // Query first input properties to get dimensions
    struct hbDNNTensorProperties props;
    ret = g_api.GetInputTensorProperties(&props, dnn_handle_, 0);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonInferenceEngine] GetInputTensorProperties failed: {}", ret);
        return false;
    }

    // Extract spatial dimensions (assume NCHW layout)
    if (props.validShape.numDimensions >= 4) {
        input_width_ = props.validShape.dimensionSize[3];
        input_height_ = props.validShape.dimensionSize[2];
    } else {
        input_width_ = config_.input_width;
        input_height_ = config_.input_height;
    }

    // Query output count and sizes
    ret = g_api.GetOutputCount(&num_outputs_, dnn_handle_);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonInferenceEngine] GetOutputCount failed: {}", ret);
        return false;
    }

    output_sizes_.resize(num_outputs_);
    for (int32_t i = 0; i < num_outputs_; ++i) {
        ret = g_api.GetOutputTensorProperties(&props, dnn_handle_, i);
        if (ret != 0) {
            LOG_ERROR_FMT("[HorizonInferenceEngine] GetOutputTensorProperties({}) failed: {}", i, ret);
            return false;
        }
        output_sizes_[i] = props.alignedByteSize;
    }

    return true;
}

bool HorizonInferenceEngine::infer(const void* input_data, size_t input_size,
                                    void* output_data, size_t output_size) {
    if (!loaded_ || !dnn_handle_) {
        LOG_ERROR("[HorizonInferenceEngine] Model not loaded");
        return false;
    }

    if (!g_api.loaded) {
        LOG_ERROR("[HorizonInferenceEngine] DNN API not loaded");
        return false;
    }

    // Allocate input tensor memory
    struct hbUCPSysMem input_mem;
    int32_t ret = g_api.Malloc(&input_mem, input_size, 0);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonInferenceEngine] Input Malloc failed: {}", ret);
        return false;
    }
    memcpy(input_mem.virAddr, input_data, input_size);

    // Setup input tensor
    struct hbDNNTensorProperties input_props;
    g_api.GetInputTensorProperties(&input_props, dnn_handle_, 0);

    struct hbDNNTensor input_tensor;
    input_tensor.sysMem = input_mem;
    input_tensor.properties = input_props;

    // Setup output tensors
    std::vector<struct hbDNNTensor> output_tensors(num_outputs_);
    std::vector<struct hbUCPSysMem> output_mems(num_outputs_);

    for (int32_t i = 0; i < num_outputs_; ++i) {
        ret = g_api.Malloc(&output_mems[i], output_sizes_[i], 0);
        if (ret != 0) {
            LOG_ERROR_FMT("[HorizonInferenceEngine] Output Malloc({}) failed: {}", i, ret);
            // Cleanup
            g_api.Free(&input_mem);
            for (int32_t j = 0; j < i; ++j) g_api.Free(&output_mems[j]);
            return false;
        }
        memset(output_mems[i].virAddr, 0, output_sizes_[i]);

        struct hbDNNTensorProperties out_props;
        g_api.GetOutputTensorProperties(&out_props, dnn_handle_, i);

        output_tensors[i].sysMem = output_mems[i];
        output_tensors[i].properties = out_props;
    }

    // Run inference
    hbUCPTaskHandle_t task_handle = nullptr;
    ret = g_api.InferV2(&task_handle, output_tensors.data(), &input_tensor, dnn_handle_);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonInferenceEngine] InferV2 failed: {}", ret);
        g_api.Free(&input_mem);
        for (auto& m : output_mems) g_api.Free(&m);
        return false;
    }

    // Wait for completion
    ret = g_api.WaitTaskDone(task_handle, 5000);  // 5s timeout
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonInferenceEngine] WaitTaskDone failed: {}", ret);
        g_api.ReleaseTask(task_handle);
        g_api.Free(&input_mem);
        for (auto& m : output_mems) g_api.Free(&m);
        return false;
    }

    // Copy output data
    size_t offset = 0;
    for (int32_t i = 0; i < num_outputs_; ++i) {
        size_t copy_size = std::min(static_cast<size_t>(output_sizes_[i]),
                                     output_size - offset);
        memcpy(static_cast<uint8_t*>(output_data) + offset,
               output_mems[i].virAddr, copy_size);
        offset += copy_size;
    }

    // Cleanup
    g_api.ReleaseTask(task_handle);
    g_api.Free(&input_mem);
    for (auto& m : output_mems) g_api.Free(&m);

    return true;
}

std::pair<int, int> HorizonInferenceEngine::getInputSize() const {
    return {input_width_, input_height_};
}

int HorizonInferenceEngine::getBatchSize() const {
    return config_.batch_size;
}

bool HorizonInferenceEngine::isAvailable() const {
    return load_horizon_dnn_lib();
}

// Register the backend
REGISTER_INFERENCE_BACKEND(InferenceBackend::HORIZON, HorizonInferenceEngine)

} // namespace hal
} // namespace ai_stream
