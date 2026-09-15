// src/hal/infer/horizon/horizon_pose_estimation.cpp
// Horizon BPU 姿态估计引擎实现
#include "horizon_pose_estimation.h"
#include "ai_stream/hal/pose_estimation_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <cstring>

extern "C" {
#include <hb_dnn.h>
#include <hb_ucp.h>
#include <hb_ucp_sys.h>
}

namespace ai_stream {
namespace hal {

static bool& get_pose_api_loaded() {
    static bool loaded = false;
    static bool tried = false;
    if (tried) return loaded;
    tried = true;

    void* h = dlopen("libdnn.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libhbrt4.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) {
        LOG_WARN("[HorizonPoseEstimation] DNN library not available");
        return loaded;
    }

    // Just verify key symbols exist
    loaded = dlsym(h, "hbDNNInitializeFromFiles") && dlsym(h, "hbDNNInferV2");
    if (!loaded) {
        LOG_ERROR("[HorizonPoseEstimation] Required DNN symbols not found");
    }
    return loaded;
}

HorizonPoseEstimationEngine::~HorizonPoseEstimationEngine() {
    if (dnn_packed_handle_) {
        void* h = dlopen("libdnn.so", RTLD_NOW | RTLD_GLOBAL);
        if (!h) h = dlopen("libhbrt4.so", RTLD_NOW | RTLD_GLOBAL);
        if (h) {
            auto release = reinterpret_cast<int32_t(*)(void*)>(dlsym(h, "hbDNNRelease"));
            if (release) release(dnn_packed_handle_);
        }
        dnn_packed_handle_ = nullptr;
    }
}

bool HorizonPoseEstimationEngine::loadModel(const PoseEstimationConfig& config) {
    config_ = config;
    LOG_INFO_FMT("[HorizonPoseEstimation] Loading model: {}", config.model_path);

    if (!get_pose_api_loaded()) {
        LOG_ERROR("[HorizonPoseEstimation] DNN SDK not available");
        return false;
    }

    void* h = dlopen("libdnn.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libhbrt4.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return false;

    auto init_files = reinterpret_cast<int32_t(*)(void**, char const**, int32_t)>(dlsym(h, "hbDNNInitializeFromFiles"));
    auto get_names = reinterpret_cast<int32_t(*)(char const***, int32_t*, void*)>(dlsym(h, "hbDNNGetModelNameList"));
    auto get_handle = reinterpret_cast<int32_t(*)(void**, void*, char const*)>(dlsym(h, "hbDNNGetModelHandle"));
    auto get_input_cnt = reinterpret_cast<int32_t(*)(int32_t*, void*)>(dlsym(h, "hbDNNGetInputCount"));
    auto get_input_props = reinterpret_cast<int32_t(*)(struct hbDNNTensorProperties*, void*, int32_t)>(dlsym(h, "hbDNNGetInputTensorProperties"));
    auto get_output_cnt = reinterpret_cast<int32_t(*)(int32_t*, void*)>(dlsym(h, "hbDNNGetOutputCount"));
    auto get_output_props = reinterpret_cast<int32_t(*)(struct hbDNNTensorProperties*, void*, int32_t)>(dlsym(h, "hbDNNGetOutputTensorProperties"));

    if (!init_files || !get_names || !get_handle) return false;

    char const* files[] = {config.model_path.c_str()};
    int32_t ret = init_files(&dnn_packed_handle_, files, 1);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonPoseEstimation] InitializeFromFiles failed: {}", ret);
        return false;
    }

    char const** names = nullptr;
    int32_t count = 0;
    ret = get_names(&names, &count, dnn_packed_handle_);
    if (ret != 0 || count <= 0) return false;

    ret = get_handle(&dnn_handle_, dnn_packed_handle_, names[0]);
    if (ret != 0) return false;

    // Query input dimensions
    if (get_input_cnt && get_input_props) {
        int32_t n_inputs = 0;
        get_input_cnt(&n_inputs, dnn_handle_);
        if (n_inputs > 0) {
            struct hbDNNTensorProperties props;
            get_input_props(&props, dnn_handle_, 0);
            if (props.validShape.numDimensions >= 4) {
                input_width_ = props.validShape.dimensionSize[3];
                input_height_ = props.validShape.dimensionSize[2];
            }
        }
    }

    // Query output size
    if (get_output_cnt && get_output_props) {
        int32_t n_outputs = 0;
        get_output_cnt(&n_outputs, dnn_handle_);
        if (n_outputs > 0) {
            struct hbDNNTensorProperties props;
            get_output_props(&props, dnn_handle_, 0);
            output_floats_per_person_ = static_cast<int>(props.alignedByteSize / sizeof(float));
        }
    }

    if (input_width_ == 0) input_width_ = config.input_width;
    if (input_height_ == 0) input_height_ = config.input_height;
    if (output_floats_per_person_ == 0) output_floats_per_person_ = 56;  // default: 4 box + 1 score + 51 kpt

    loaded_ = true;
    LOG_INFO_FMT("[HorizonPoseEstimation] Model loaded: {}x{}, {} floats/person",
                 input_width_, input_height_, output_floats_per_person_);
    return true;
}

bool HorizonPoseEstimationEngine::inferHost(const float* input_nchw, int num_persons,
                                             std::vector<float>& output_host) {
    if (!loaded_ || !dnn_handle_) {
        LOG_ERROR("[HorizonPoseEstimation] Model not loaded");
        return false;
    }

    // Simplified: run batch inference and copy output
    // Full implementation would follow the same pattern as horizon_inference_engine.cpp
    LOG_WARN("[HorizonPoseEstimation] inferHost: full BPU inference not yet implemented, returning mock");
    output_host.resize(num_persons * output_floats_per_person_, 0.0f);
    return true;
}

std::pair<int, int> HorizonPoseEstimationEngine::getInputSize() const {
    return {input_width_, input_height_};
}

bool HorizonPoseEstimationEngine::isAvailable() const {
    return get_pose_api_loaded();
}

size_t HorizonPoseEstimationEngine::getOutputFloatsPerPerson() const {
    return output_floats_per_person_;
}

REGISTER_POSE_ESTIMATION_BACKEND(PoseEstimationBackend::HORIZON, HorizonPoseEstimationEngine)

} // namespace hal
} // namespace ai_stream
