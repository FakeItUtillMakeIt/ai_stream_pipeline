// src/hal/infer/intel/openvino_detection_engine.cpp
#include "openvino_detection_engine.h"
#include "ai_stream/hal/detection_inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace ai_stream {
namespace hal {
namespace {

constexpr const char* kBoxesName = "det_boxes";
constexpr const char* kScoresName = "det_scores";
constexpr const char* kClassesName = "det_classes";
constexpr const char* kBatchIdsName = "det_batch_ids";
constexpr const char* kNumDetsName = "det_num_dets";

constexpr const char* kKnownOutputs[] = {
    kBoxesName, kScoresName, kClassesName, kBatchIdsName, kNumDetsName
};

void logStatus(ov_status_e status, const char* what) {
    LOG_ERROR_FMT("[OpenVINO] {} failed: {} ({})", what,
                  static_cast<int>(status), ov_get_error_info(status));
    const char* msg = ov_get_last_err_msg();
    if (msg && *msg) LOG_ERROR_FMT("[OpenVINO]   {}", msg);
}

std::string pickDevice(ov_core_t* core) {
    const char* env = std::getenv("AI_STREAM_OPENVINO_DEVICE");
    if (env && *env) return env;

    ov_available_devices_t devices{};
    if (ov_core_get_available_devices(core, &devices) == OK) {
        bool has_gpu0 = false, has_gpu = false, has_npu = false, has_cpu = false;
        for (size_t i = 0; i < devices.size; ++i) {
            const char* d = devices.devices[i];
            if (!d) continue;
            if (std::strcmp(d, "GPU.0") == 0) has_gpu0 = true;
            else if (std::strncmp(d, "GPU", 3) == 0) has_gpu = true;
            else if (std::strncmp(d, "NPU", 3) == 0) has_npu = true;
            else if (std::strncmp(d, "CPU", 3) == 0) has_cpu = true;
        }
        ov_available_devices_free(&devices);
        // Intel 核显优先；没有 GPU 时退 NPU/CPU
        if (has_gpu0) return "GPU.0";
        if (has_gpu) return "GPU";
        if (has_npu) return "NPU";
        if (has_cpu) return "CPU";
    }
    return "CPU";
}

bool portName(ov_output_const_port_t* port, std::string& out) {
    if (!port) return false;
    char* name = nullptr;
    if (ov_port_get_any_name(port, &name) != OK || !name) return false;
    out = name;
    ov_free(name);
    return true;
}

} // namespace

OpenVinoDetectionEngine::~OpenVinoDetectionEngine() {
    cleanup();
}

void OpenVinoDetectionEngine::cleanup() {
    if (request_) ov_infer_request_free(request_);
    if (compiled_) ov_compiled_model_free(compiled_);
    if (model_) ov_model_free(model_);
    if (core_) ov_core_free(core_);
    request_ = nullptr;
    compiled_ = nullptr;
    model_ = nullptr;
    core_ = nullptr;
    output_index_.clear();
    input_names_.clear();
    output_names_.clear();
    external_outputs_.clear();
    boxes_.clear();
    scores_.clear();
    classes_.clear();
    batch_ids_.clear();
    num_dets_ = 0;
}

bool OpenVinoDetectionEngine::loadModel(const DetectionInferenceConfig& config) {
    cleanup();
    config_ = config;

    if (ov_core_create(&core_) != OK || !core_) {
        LOG_ERROR("[OpenVINO] Failed to create core");
        cleanup();
        return false;
    }

    if (ov_core_read_model(core_, config.model_path.c_str(), nullptr, &model_) != OK ||
        !model_) {
        LOG_ERROR_FMT("[OpenVINO] Failed to read model '{}' (ONNX/IR only; .engine is TensorRT)",
                      config.model_path);
        const char* msg = ov_get_last_err_msg();
        if (msg && *msg) LOG_ERROR_FMT("[OpenVINO]   {}", msg);
        cleanup();
        return false;
    }

    device_name_ = pickDevice(core_);
    ov_status_e status = ov_core_compile_model(core_, model_, device_name_.c_str(),
                                               0, &compiled_);
    if (status != OK || !compiled_) {
        logStatus(status, ("compile_model " + device_name_).c_str());
        if (device_name_ != "CPU") {
            LOG_WARN_FMT("[OpenVINO] Falling back to CPU (GPU/NPU runtime unavailable)");
            device_name_ = "CPU";
            status = ov_core_compile_model(core_, model_, device_name_.c_str(),
                                           0, &compiled_);
        }
    }
    if (status != OK || !compiled_) {
        logStatus(status, "compile_model CPU");
        cleanup();
        return false;
    }

    if (ov_compiled_model_create_infer_request(compiled_, &request_) != OK || !request_) {
        LOG_ERROR("[OpenVINO] Failed to create infer request");
        cleanup();
        return false;
    }

    // 输入名列表
    size_t input_count = 0;
    if (ov_model_inputs_size(model_, &input_count) == OK) {
        for (size_t i = 0; i < input_count; ++i) {
            ov_output_const_port_t* port = nullptr;
            if (ov_model_const_input_by_index(model_, i, &port) != OK || !port) continue;
            std::string name;
            if (portName(port, name)) input_names_.push_back(name);
            ov_output_const_port_free(port);
        }
    }
    if (!input_names_.empty()) input_name_ = input_names_.front();

    if (!parseOutputIndices()) {
        cleanup();
        return false;
    }

    LOG_INFO_FMT("[OpenVINO] Loaded '{}' on {} ({} outputs)",
                 config.model_path, device_name_, output_names_.size());
    return true;
}

bool OpenVinoDetectionEngine::parseOutputIndices() {
    size_t count = 0;
    if (ov_compiled_model_outputs_size(compiled_, &count) != OK || count == 0) {
        LOG_ERROR("[OpenVINO] Model has no outputs");
        return false;
    }

    std::vector<std::string> names(count);
    for (size_t i = 0; i < count; ++i) {
        ov_output_const_port_t* port = nullptr;
        if (ov_compiled_model_output_by_index(compiled_, i, &port) != OK || !port) {
            continue;
        }
        portName(port, names[i]);
        ov_output_const_port_free(port);
    }
    output_names_ = names;

    for (const char* expected : kKnownOutputs) {
        for (size_t i = 0; i < names.size(); ++i) {
            if (names[i] == expected) {
                output_index_[expected] = i;
                break;
            }
        }
    }

    // 名字不匹配但正好 5 个输出：按标准顺序兜底（部分导出工具会重命名）
    if (output_index_.size() != 5 && count == 5) {
        LOG_WARN("[OpenVINO] Output names do not match contract, using positional order");
        for (size_t i = 0; i < 5; ++i) {
            output_index_[kKnownOutputs[i]] = i;
        }
    }

    if (output_index_.size() != 5) {
        std::string joined;
        for (const auto& n : names) {
            if (!joined.empty()) joined += ",";
            joined += n;
        }
        LOG_ERROR_FMT("[OpenVINO] Expected 5 detection outputs, got {} (model outputs: {})",
                      count, joined);
        return false;
    }
    return true;
}

bool OpenVinoDetectionEngine::setInputTensor(const std::string& name, void* gpu_ptr) {
    input_name_ = name.empty() ? input_name_ : name;
    input_ptr_ = gpu_ptr;
    return input_ptr_ != nullptr;
}

bool OpenVinoDetectionEngine::setOutputTensor(const std::string& name, void* gpu_ptr) {
    external_outputs_[name] = gpu_ptr;
    return true;
}

std::vector<std::string> OpenVinoDetectionEngine::getInputNames() const {
    return input_names_;
}

std::vector<std::string> OpenVinoDetectionEngine::getOutputNames() const {
    return output_names_;
}

std::string OpenVinoDetectionEngine::getBackendName() const {
    return "OpenVINO (" + device_name_ + ")";
}

bool OpenVinoDetectionEngine::infer() {
    if (!request_ || !compiled_ || !input_ptr_) {
        LOG_ERROR("[OpenVINO] Infer called before model/input is ready");
        return false;
    }

    const int w = config_.input_width > 0 ? config_.input_width : 640;
    const int h = config_.input_height > 0 ? config_.input_height : 640;
    int64_t dims[4] = {1, 3, h, w};

    ov_shape_t shape{};
    if (ov_shape_create(4, dims, &shape) != OK) {
        LOG_ERROR("[OpenVINO] Failed to create input shape");
        return false;
    }

    ov_tensor_t* tensor = nullptr;
    ov_status_e status = ov_tensor_create_from_host_ptr(
        F32, shape, const_cast<void*>(input_ptr_), &tensor);
    ov_shape_free(&shape);
    if (status != OK || !tensor) {
        logStatus(status, "create input tensor");
        return false;
    }

    status = ov_infer_request_set_input_tensor(request_, tensor);
    if (status != OK) {
        logStatus(status, "set_input_tensor");
        ov_tensor_free(tensor);
        return false;
    }

    status = ov_infer_request_infer(request_);
    ov_tensor_free(tensor);
    if (status != OK) {
        logStatus(status, "infer");
        return false;
    }

    if (!copyOutputs()) return false;
    copyToExternalBuffers();
    return true;
}

namespace {

bool readF32(ov_infer_request_t* req, size_t index, std::vector<float>& out) {
    ov_tensor_t* t = nullptr;
    if (ov_infer_request_get_output_tensor_by_index(req, index, &t) != OK || !t) {
        out.clear();
        return false;
    }
    size_t n = 0;
    ov_tensor_get_size(t, &n);
    void* data = nullptr;
    ov_element_type_e type = UNDEFINED;
    ov_tensor_get_element_type(t, &type);
    ov_tensor_data(t, &data);
    out.clear();
    if (n > 0 && data && type == F32) {
        const float* p = static_cast<const float*>(data);
        out.assign(p, p + n);
    }
    ov_tensor_free(t);
    return true;
}

bool readI64(ov_infer_request_t* req, size_t index, std::vector<int64_t>& out) {
    ov_tensor_t* t = nullptr;
    if (ov_infer_request_get_output_tensor_by_index(req, index, &t) != OK || !t) {
        out.clear();
        return false;
    }
    size_t n = 0;
    ov_tensor_get_size(t, &n);
    void* data = nullptr;
    ov_element_type_e type = UNDEFINED;
    ov_tensor_get_element_type(t, &type);
    ov_tensor_data(t, &data);
    out.clear();
    if (n > 0 && data) {
        if (type == I64) {
            const int64_t* p = static_cast<const int64_t*>(data);
            out.assign(p, p + n);
        } else if (type == I32) {
            const int32_t* p = static_cast<const int32_t*>(data);
            out.assign(p, p + n);
        } else if (type == F32) {
            const float* p = static_cast<const float*>(data);
            out.resize(n);
            for (size_t i = 0; i < n; ++i) out[i] = static_cast<int64_t>(p[i]);
        }
    }
    ov_tensor_free(t);
    return true;
}

} // namespace

bool OpenVinoDetectionEngine::copyOutputs() {
    std::vector<int64_t> num;
    if (!readI64(request_, output_index_.at(kNumDetsName), num) || num.empty()) {
        LOG_ERROR("[OpenVINO] det_num_dets missing");
        return false;
    }
    int64_t dets = std::max<int64_t>(0, num.front());
    const int64_t cap = config_.max_detections > 0 ? config_.max_detections : 200;
    if (dets > cap) dets = cap;

    std::vector<float> boxes, scores;
    std::vector<int64_t> classes, batch_ids;
    readF32(request_, output_index_.at(kBoxesName), boxes);
    readF32(request_, output_index_.at(kScoresName), scores);
    readI64(request_, output_index_.at(kClassesName), classes);
    readI64(request_, output_index_.at(kBatchIdsName), batch_ids);

    // 以张量实际容量为准（NMS 后 det_num_dets 与实际行数一致，这里做防御性截断）
    int64_t actual = dets;
    actual = std::min<int64_t>(actual, boxes.size() / 4);
    actual = std::min<int64_t>(actual, scores.size());
    actual = std::min<int64_t>(actual, classes.size());
    actual = std::min<int64_t>(actual, batch_ids.size());
    if (actual < 0) actual = 0;

    num_dets_ = actual;
    boxes_.assign(boxes.begin(), boxes.begin() + actual * 4);
    scores_.assign(scores.begin(), scores.begin() + actual);
    classes_.assign(classes.begin(), classes.begin() + actual);
    batch_ids_.assign(batch_ids.begin(), batch_ids.begin() + actual);
    return true;
}

bool OpenVinoDetectionEngine::copyToExternalBuffers() {
    auto copyBuf = [&](const char* name, const void* src, size_t bytes) {
        auto it = external_outputs_.find(name);
        if (it == external_outputs_.end() || !it->second || bytes == 0) return;
        std::memcpy(it->second, src, bytes);
    };
    copyBuf(kNumDetsName, &num_dets_, sizeof(num_dets_));
    copyBuf(kBoxesName, boxes_.data(), boxes_.size() * sizeof(float));
    copyBuf(kScoresName, scores_.data(), scores_.size() * sizeof(float));
    copyBuf(kClassesName, classes_.data(), classes_.size() * sizeof(int64_t));
    copyBuf(kBatchIdsName, batch_ids_.data(), batch_ids_.size() * sizeof(int64_t));
    return true;
}

void* OpenVinoDetectionEngine::getOutputTensor(const std::string& name) {
    if (name == kNumDetsName) return &num_dets_;
    if (name == kBoxesName) return boxes_.data();
    if (name == kScoresName) return scores_.data();
    if (name == kClassesName) return classes_.data();
    if (name == kBatchIdsName) return batch_ids_.data();
    return nullptr;
}

size_t OpenVinoDetectionEngine::getOutputTensorSize(const std::string& name) const {
    if (name == kNumDetsName) return sizeof(int64_t);
    if (name == kBoxesName) return boxes_.size() * sizeof(float);
    if (name == kScoresName) return scores_.size() * sizeof(float);
    if (name == kClassesName) return classes_.size() * sizeof(int64_t);
    if (name == kBatchIdsName) return batch_ids_.size() * sizeof(int64_t);
    return 0;
}

REGISTER_DETECTION_INFERENCE_BACKEND(DetectionBackend::OPENVINO, OpenVinoDetectionEngine)

} // namespace hal
} // namespace ai_stream
