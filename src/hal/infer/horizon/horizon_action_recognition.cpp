// src/hal/infer/horizon/horizon_action_recognition.cpp
// Horizon BPU 动作识别引擎实现
#include "horizon_action_recognition.h"
#include "ai_stream/hal/action_recognition_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <cstring>
#include <algorithm>

extern "C" {
#include <hb_dnn.h>
#include <hb_ucp.h>
#include <hb_ucp_sys.h>
}

namespace ai_stream {
namespace hal {

static bool& get_action_api_loaded() {
    static bool loaded = false;
    static bool tried = false;
    if (tried) return loaded;
    tried = true;

    void* h = dlopen("libdnn.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libhbrt4.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return loaded;

    loaded = dlsym(h, "hbDNNInitializeFromFiles") && dlsym(h, "hbDNNInferV2");
    return loaded;
}

HorizonActionRecognitionEngine::~HorizonActionRecognitionEngine() {
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

bool HorizonActionRecognitionEngine::loadModel(const ActionRecognitionConfig& config) {
    config_ = config;
    LOG_INFO_FMT("[HorizonActionRecognition] Loading model: {}", config.model_path);

    if (!get_action_api_loaded()) {
        LOG_ERROR("[HorizonActionRecognition] DNN SDK not available");
        return false;
    }

    void* h = dlopen("libdnn.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) h = dlopen("libhbrt4.so", RTLD_NOW | RTLD_GLOBAL);
    if (!h) return false;

    auto init_files = reinterpret_cast<int32_t(*)(void**, char const**, int32_t)>(dlsym(h, "hbDNNInitializeFromFiles"));
    auto get_names = reinterpret_cast<int32_t(*)(char const***, int32_t*, void*)>(dlsym(h, "hbDNNGetModelNameList"));
    auto get_handle = reinterpret_cast<int32_t(*)(void**, void*, char const*)>(dlsym(h, "hbDNNGetModelHandle"));

    if (!init_files || !get_names || !get_handle) return false;

    char const* files[] = {config.model_path.c_str()};
    int32_t ret = init_files(&dnn_packed_handle_, files, 1);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonActionRecognition] InitializeFromFiles failed: {}", ret);
        return false;
    }

    char const** names = nullptr;
    int32_t count = 0;
    ret = get_names(&names, &count, dnn_packed_handle_);
    if (ret != 0 || count <= 0) return false;

    ret = get_handle(&dnn_handle_, dnn_packed_handle_, names[0]);
    if (ret != 0) return false;

    loaded_ = true;
    LOG_INFO("[HorizonActionRecognition] Model loaded");
    return true;
}

bool HorizonActionRecognitionEngine::infer(const uint8_t* clip_data, size_t clip_size,
                                            ActionResult& result) {
    if (!loaded_ || !dnn_handle_) {
        LOG_ERROR("[HorizonActionRecognition] Model not loaded");
        return false;
    }

    // Simplified: full BPU inference would follow the same pattern
    LOG_WARN("[HorizonActionRecognition] infer: full BPU inference not yet implemented, returning mock");

    // Mock result
    result.action_id = 0;
    result.confidence = 0.0f;
    if (!config_.action_labels.empty()) {
        result.action_label = config_.action_labels[0];
    }
    result.scores.resize(config_.action_labels.size(), 0.0f);
    return true;
}

std::pair<int, int> HorizonActionRecognitionEngine::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int HorizonActionRecognitionEngine::getNumFrames() const {
    return config_.num_frames;
}

bool HorizonActionRecognitionEngine::isAvailable() const {
    return get_action_api_loaded();
}

REGISTER_ACTION_RECOGNITION_BACKEND(ActionRecognitionBackend::HORIZON, HorizonActionRecognitionEngine)

} // namespace hal
} // namespace ai_stream
