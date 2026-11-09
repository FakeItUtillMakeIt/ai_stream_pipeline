// src/hal/infer/ascend/ascend_action_recognition.cpp
// Ascend OM 动作识别引擎——ACL aclmdl + softmax 后处理
#include "ascend_action_recognition.h"
#include "ai_stream/hal/action_recognition_factory.h"
#include "ascend/ascend_acl.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace ai_stream {
namespace hal {
namespace {

size_t typeSize(aclDataType type) {
    switch (type) {
        case ACL_FLOAT:   return 4;
        case ACL_FLOAT16: return 2;
        case ACL_INT64:   return 8;
        case ACL_INT32:   return 4;
        case ACL_INT16:   return 2;
        case ACL_INT8:    return 1;
        case ACL_UINT8:   return 1;
        default:          return 4;
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

AscendActionRecognition::~AscendActionRecognition() {
    cleanup();
}

bool AscendActionRecognition::isAvailable() const {
    return ::access("/dev/davinci0", R_OK | W_OK) == 0;
}

std::string AscendActionRecognition::getBackendName() const {
    return soc_name_.empty() ? "Ascend OM Action Recognition (Ascend NPU)"
                             : ("Ascend OM Action Recognition (" + soc_name_ + ")");
}

bool AscendActionRecognition::initAcl() {
    if (!ensureAscendAclInitialized()) {
        LOG_ERROR("[AscendActionRecognition] aclInit failed");
        return false;
    }
    uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS || count == 0) {
        LOG_ERROR("[AscendActionRecognition] No Ascend device found");
        return false;
    }
    int device = config_.device_id >= 0 ? config_.device_id : 0;
    if (static_cast<uint32_t>(device) >= count) device = 0;
    if (aclrtSetDevice(device) != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendActionRecognition] aclrtSetDevice({}) failed", device);
        return false;
    }
    config_.device_id = device;
    const char* soc = aclrtGetSocName();
    if (soc) soc_name_ = soc;
    return true;
}

bool AscendActionRecognition::loadModel(const ActionRecognitionConfig& config) {
    cleanup();
    config_ = config;

    if (!initAcl()) {
        cleanup();
        return false;
    }

    aclError ret = aclmdlLoadFromFile(config.model_path.c_str(), &model_id_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendActionRecognition] aclmdlLoadFromFile failed: {} ({}); "
                      "only .om is supported, convert with ATC",
                      static_cast<int>(ret), config.model_path);
        cleanup();
        return false;
    }
    loaded_ = true;

    desc_ = aclmdlCreateDesc();
    if (!desc_ || aclmdlGetDesc(desc_, model_id_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendActionRecognition] aclmdlGetDesc failed");
        cleanup();
        return false;
    }
    if (!parseIo()) {
        cleanup();
        return false;
    }

    LOG_INFO_FMT("[AscendActionRecognition] Loaded '{}' on {} (input={}B, output={}B, classes={})",
                 config.model_path, getBackendName(), input_size_, output_size_, output_elements_);
    return true;
}

bool AscendActionRecognition::parseIo() {
    const size_t in_count = aclmdlGetNumInputs(desc_);
    const size_t out_count = aclmdlGetNumOutputs(desc_);
    if (in_count != 1 || out_count == 0) {
        LOG_ERROR_FMT("[AscendActionRecognition] Expect 1 input, got {} inputs / {} outputs",
                      in_count, out_count);
        return false;
    }

    input_size_ = aclmdlGetInputSizeByIndex(desc_, 0);
    if (input_size_ == 0 || aclmdlGetInputDataType(desc_, 0) != ACL_FLOAT) {
        LOG_ERROR("[AscendActionRecognition] Input must be FP32");
        return false;
    }
    aclmdlIODims in_dims{};
    if (aclmdlGetInputDims(desc_, 0, &in_dims) == ACL_SUCCESS && in_dims.dimCount == 5) {
        // VideoMAE: [1, T, 3, H, W]
        const int frames = static_cast<int>(in_dims.dims[1]);
        const int height = static_cast<int>(in_dims.dims[3]);
        const int width = static_cast<int>(in_dims.dims[4]);
        if (frames > 0) config_.num_frames = frames;
        if (height > 0) config_.input_height = height;
        if (width > 0) config_.input_width = width;
    }

    void* d_in = nullptr;
    if (aclrtMalloc(&d_in, input_size_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !d_in) {
        LOG_ERROR("[AscendActionRecognition] aclrtMalloc input failed");
        return false;
    }
    d_input_ = d_in;
    input_buf_ = aclCreateDataBuffer(d_input_, input_size_);
    input_ds_ = aclmdlCreateDataset();
    if (!input_buf_ || !input_ds_ ||
        aclmdlAddDatasetBuffer(input_ds_, input_buf_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendActionRecognition] create input dataset failed");
        return false;
    }

    output_ds_ = aclmdlCreateDataset();
    if (!output_ds_) {
        LOG_ERROR("[AscendActionRecognition] create output dataset failed");
        return false;
    }
    // 动作模型通常只有 1 个 logits 输出；若多个，取第一个（node 只消费一个 ActionResult）
    output_size_ = aclmdlGetOutputSizeByIndex(desc_, 0);
    aclDataType out_type = aclmdlGetOutputDataType(desc_, 0);
    if (output_size_ == 0) {
        LOG_ERROR("[AscendActionRecognition] output size is zero");
        return false;
    }
    aclmdlIODims out_dims{};
    if (aclmdlGetOutputDims(desc_, 0, &out_dims) == ACL_SUCCESS) {
        output_elements_ = elementsFromDims(out_dims);
    }
    if (output_elements_ == 0) {
        output_elements_ = output_size_ / typeSize(out_type);
    }
    if (out_type != ACL_FLOAT) {
        LOG_WARN_FMT("[AscendActionRecognition] output type {} (not FP32), will convert",
                     static_cast<int>(out_type));
    }

    void* d_out = nullptr;
    if (aclrtMalloc(&d_out, output_size_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !d_out) {
        LOG_ERROR("[AscendActionRecognition] aclrtMalloc output failed");
        return false;
    }
    d_output_ = d_out;
    output_buf_ = aclCreateDataBuffer(d_output_, output_size_);
    if (!output_buf_ ||
        aclmdlAddDatasetBuffer(output_ds_, output_buf_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendActionRecognition] add output dataset failed");
        return false;
    }
    return true;
}

bool AscendActionRecognition::inferPreprocessed(const float* input_nchw,
                                                size_t size_bytes,
                                                ActionResult& result) {
    if (!loaded_ || !input_nchw || !input_ds_ || !output_ds_) {
        LOG_ERROR("[AscendActionRecognition] inferPreprocessed before model ready");
        return false;
    }
    if (size_bytes < input_size_) {
        LOG_ERROR_FMT("[AscendActionRecognition] input too small: {} < {}",
                      size_bytes, input_size_);
        return false;
    }

    aclError ret = aclrtSetDevice(config_.device_id);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendActionRecognition] set device failed: {}", static_cast<int>(ret));
        return false;
    }
    ret = aclrtMemcpy(d_input_, input_size_, input_nchw, input_size_,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendActionRecognition] H2D failed: {}", static_cast<int>(ret));
        return false;
    }
    ret = aclmdlExecute(model_id_, input_ds_, output_ds_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendActionRecognition] aclmdlExecute failed: {}", static_cast<int>(ret));
        return false;
    }

    std::vector<float> logits(output_elements_);
    const size_t copy_bytes = std::min(output_size_, output_elements_ * sizeof(float));
    ret = aclrtMemcpy(logits.data(), copy_bytes, d_output_, copy_bytes,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendActionRecognition] D2H failed: {}", static_cast<int>(ret));
        return false;
    }

    result = postprocess(logits.data(), output_elements_);
    return true;
}

bool AscendActionRecognition::infer(const uint8_t* clip_data, size_t clip_size,
                                    ActionResult& result) {
    // 兼容旧接口：clip_data 实际是已归一化 float NCHW（与 TensorRT 的
    // inferPreprocessed 路径一致）；byte 版本通常来自 uint8 clip，未在本项目使用。
    return inferPreprocessed(reinterpret_cast<const float*>(clip_data), clip_size, result);
}

ActionResult AscendActionRecognition::postprocess(const float* logits, size_t count) const {
    ActionResult result;
    if (!logits || count == 0) return result;

    const size_t num_classes = std::min<size_t>(
        count, config_.action_labels.empty() ? count : config_.action_labels.size());
    float max_val = *std::max_element(logits, logits + num_classes);
    result.scores.resize(num_classes);
    float sum = 0.0f;
    for (size_t i = 0; i < num_classes; ++i) {
        result.scores[i] = std::exp(logits[i] - max_val);
        sum += result.scores[i];
    }
    if (sum > 0.0f) {
        for (auto& v : result.scores) v /= sum;
    }

    size_t best = 0;
    for (size_t i = 1; i < num_classes; ++i) {
        if (result.scores[i] > result.scores[best]) best = i;
    }
    result.action_id = static_cast<int>(best);
    result.confidence = result.scores[best];
    if (best < config_.action_labels.size()) {
        result.action_label = config_.action_labels[best];
    } else {
        result.action_label = std::to_string(best);
    }

    // top-k（最多 3 个）
    std::vector<size_t> order(num_classes);
    for (size_t i = 0; i < num_classes; ++i) order[i] = i;
    std::partial_sort(order.begin(), order.begin() + std::min<size_t>(3, num_classes),
                      order.end(),
                      [&](size_t a, size_t b) { return result.scores[a] > result.scores[b]; });
    for (size_t i = 0; i < std::min<size_t>(3, num_classes); ++i) {
        const size_t idx = order[i];
        const std::string label = idx < config_.action_labels.size()
                                      ? config_.action_labels[idx]
                                      : std::to_string(idx);
        result.top_k.emplace_back(label, result.scores[idx]);
    }
    return result;
}

std::pair<int, int> AscendActionRecognition::getInputSize() const {
    return {config_.input_width, config_.input_height};
}

int AscendActionRecognition::getNumFrames() const {
    return config_.num_frames;
}

void AscendActionRecognition::cleanup() {
    if (input_ds_) {
        aclmdlDestroyDataset(input_ds_);
        input_ds_ = nullptr;
    }
    if (input_buf_) {
        aclDestroyDataBuffer(input_buf_);
        input_buf_ = nullptr;
    }
    if (d_input_) {
        aclrtFree(d_input_);
        d_input_ = nullptr;
    }
    if (output_ds_) {
        aclmdlDestroyDataset(output_ds_);
        output_ds_ = nullptr;
    }
    if (output_buf_) {
        aclDestroyDataBuffer(output_buf_);
        output_buf_ = nullptr;
    }
    if (d_output_) {
        aclrtFree(d_output_);
        d_output_ = nullptr;
    }
    if (desc_) {
        aclmdlDestroyDesc(desc_);
        desc_ = nullptr;
    }
    if (loaded_) {
        aclmdlUnload(model_id_);
        loaded_ = false;
    }
    input_size_ = output_size_ = output_elements_ = 0;
}

#ifdef WITH_ASCEND
REGISTER_ACTION_RECOGNITION_BACKEND(ActionRecognitionBackend::ASCEND, AscendActionRecognition)
#endif

} // namespace hal
} // namespace ai_stream
