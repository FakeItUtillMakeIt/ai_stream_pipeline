// src/hal/backend_diagnostics.cpp
// HAL 后端诊断实现——查询各工厂的运行时可用后端
#include "ai_stream/hal/backend_diagnostics.h"

#include "ai_stream/hal/inference_engine_factory.h"
#include "ai_stream/hal/detection_inference_engine_factory.h"
#include "ai_stream/hal/pose_estimation_factory.h"
#include "ai_stream/hal/action_recognition_factory.h"
#include "ai_stream/hal/image_accelerator_factory.h"
#include "ai_stream/hal/video_codec_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <nlohmann/json.hpp>
#include <string>
#include <utility>
#include <vector>

namespace ai_stream {
namespace hal {
namespace {

const char* inferenceBackendName(InferenceBackend b) {
    switch (b) {
        case InferenceBackend::AUTO:     return "auto";
        case InferenceBackend::TENSORRT: return "tensorrt";
        case InferenceBackend::RKNN:     return "rknn";
        case InferenceBackend::ASCEND:   return "ascend";
        case InferenceBackend::CPU:      return "cpu";
    }
    return "unknown";
}

const char* detectionBackendName(DetectionBackend b) {
    switch (b) {
        case DetectionBackend::AUTO:     return "auto";
        case DetectionBackend::TENSORRT: return "tensorrt";
        case DetectionBackend::RKNN:     return "rknn";
        case DetectionBackend::ASCEND:   return "ascend";
        case DetectionBackend::CPU:      return "cpu";
    }
    return "unknown";
}

const char* poseBackendName(PoseEstimationBackend b) {
    switch (b) {
        case PoseEstimationBackend::AUTO:     return "auto";
        case PoseEstimationBackend::TENSORRT: return "tensorrt";
        case PoseEstimationBackend::RKNN:     return "rknn";
        case PoseEstimationBackend::ASCEND:   return "ascend";
        case PoseEstimationBackend::CPU:      return "cpu";
    }
    return "unknown";
}

const char* actionBackendName(ActionRecognitionBackend b) {
    switch (b) {
        case ActionRecognitionBackend::AUTO:     return "auto";
        case ActionRecognitionBackend::TENSORRT: return "tensorrt";
        case ActionRecognitionBackend::RKNN:     return "rknn";
        case ActionRecognitionBackend::ASCEND:   return "ascend";
        case ActionRecognitionBackend::CPU:      return "cpu";
    }
    return "unknown";
}

const char* imageAccelBackendName(ImageAcceleratorBackend b) {
    switch (b) {
        case ImageAcceleratorBackend::AUTO: return "auto";
        case ImageAcceleratorBackend::NPP:  return "npp";
        case ImageAcceleratorBackend::RGA:  return "rga";
        case ImageAcceleratorBackend::DVPP: return "dvpp";
        case ImageAcceleratorBackend::CPU:  return "cpu";
    }
    return "unknown";
}

const char* videoCodecBackendName(VideoCodecBackend b) {
    switch (b) {
        case VideoCodecBackend::AUTO:   return "auto";
        case VideoCodecBackend::NVDEC:  return "nvdec";
        case VideoCodecBackend::MPP:    return "mpp";
        case VideoCodecBackend::DVPP:   return "dvpp";
        case VideoCodecBackend::FFMPEG: return "ffmpeg";
    }
    return "unknown";
}

template <typename EnumT, typename ToStringFn>
nlohmann::json toJson(const std::vector<std::pair<EnumT, std::string>>& backends,
                      ToStringFn to_str) {
    auto arr = nlohmann::json::array();
    for (const auto& [type, name] : backends) {
        arr.push_back({{"backend", to_str(type)}, {"name", name}});
    }
    return arr;
}

template <typename EnumT, typename ToStringFn>
std::string toLogString(const std::vector<std::pair<EnumT, std::string>>& backends,
                        ToStringFn to_str) {
    if (backends.empty()) return "(none available)";
    std::string s;
    for (const auto& [type, name] : backends) {
        if (!s.empty()) s += ", ";
        s += std::string(to_str(type)) + " (" + name + ")";
    }
    return s;
}

} // namespace

void logAvailableBackends() {
    LOG_INFO_FMT("[Backends] inference: {}",
                 toLogString(InferenceEngineFactory::instance().getAvailableBackends(), &inferenceBackendName));
    LOG_INFO_FMT("[Backends] detection: {}",
                 toLogString(DetectionInferenceEngineFactory::instance().getAvailableBackends(), &detectionBackendName));
    LOG_INFO_FMT("[Backends] pose_estimation: {}",
                 toLogString(PoseEstimationFactory::instance().getAvailableBackends(), &poseBackendName));
    LOG_INFO_FMT("[Backends] action_recognition: {}",
                 toLogString(ActionRecognitionFactory::instance().getAvailableBackends(), &actionBackendName));
    LOG_INFO_FMT("[Backends] image_accelerator: {}",
                 toLogString(ImageAcceleratorFactory::instance().getAvailableBackends(), &imageAccelBackendName));
    LOG_INFO_FMT("[Backends] video_codec: {}",
                 toLogString(VideoCodecFactory::instance().getAvailableBackends(), &videoCodecBackendName));
}

std::string availableBackendsJson() {
    nlohmann::json j;
    j["inference"] = toJson(InferenceEngineFactory::instance().getAvailableBackends(), &inferenceBackendName);
    j["detection"] = toJson(DetectionInferenceEngineFactory::instance().getAvailableBackends(), &detectionBackendName);
    j["pose_estimation"] = toJson(PoseEstimationFactory::instance().getAvailableBackends(), &poseBackendName);
    j["action_recognition"] = toJson(ActionRecognitionFactory::instance().getAvailableBackends(), &actionBackendName);
    j["image_accelerator"] = toJson(ImageAcceleratorFactory::instance().getAvailableBackends(), &imageAccelBackendName);
    j["video_codec"] = toJson(VideoCodecFactory::instance().getAvailableBackends(), &videoCodecBackendName);
    return j.dump();
}

} // namespace hal
} // namespace ai_stream
