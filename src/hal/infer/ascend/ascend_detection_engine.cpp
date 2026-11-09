// src/hal/infer/ascend/ascend_detection_engine.cpp
#include "ascend_detection_engine.h"
#include "ai_stream/hal/detection_inference_engine_factory.h"
#include "ascend/ascend_acl.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>
#include <cstring>
#include <unistd.h>

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

size_t typeSize(aclDataType type) {
    switch (type) {
        case ACL_FLOAT:  return 4;
        case ACL_FLOAT16: return 2;
        case ACL_INT64:  return 8;
        case ACL_UINT64: return 8;
        case ACL_INT32:  return 4;
        case ACL_UINT32: return 4;
        case ACL_INT16:  return 2;
        case ACL_UINT16: return 2;
        case ACL_INT8:   return 1;
        case ACL_UINT8:  return 1;
        default:         return 4;
    }
}

float halfToFloat(uint16_t h) {
    // 标准 IEEE 754 half -> float
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x03ff;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) {
            f = sign;
        } else {
            exp = 127 - 15 + 1;
            while ((mant & 0x0400) == 0) { mant <<= 1; --exp; }
            mant &= 0x03ff;
            f = sign | (exp << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        f = sign | 0x7f800000 | (mant << 13);
    } else {
        f = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
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

// ACL 的 device context 是线程相关的，每个线程首次使用前要 setDevice。

} // namespace

AscendDetectionEngine::~AscendDetectionEngine() {
    cleanup();
}

bool AscendDetectionEngine::isAvailable() const {
    // 驱动/设备节点可访问即认为后端可用；实际 aclInit 在 loadModel 里做。
    return ::access("/dev/davinci0", R_OK | W_OK) == 0;
}

std::string AscendDetectionEngine::getBackendName() const {
    return soc_name_.empty() ? "Ascend ACL" : ("Ascend ACL (" + soc_name_ + ")");
}

bool AscendDetectionEngine::initAcl() {
    if (!ensureAscendAclInitialized()) {
        LOG_ERROR("[AscendDetection] aclInit failed");
        return false;
    }
    uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS || count == 0) {
        LOG_ERROR("[AscendDetection] No Ascend device found");
        return false;
    }
    if (device_id_ < 0 || static_cast<uint32_t>(device_id_) >= count) {
        device_id_ = 0;
    }
    aclError ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendDetection] aclrtSetDevice({}) failed: {}", device_id_, static_cast<int>(ret));
        return false;
    }
    acl_device_set_ = true;
    const char* soc = aclrtGetSocName();
    if (soc) soc_name_ = soc;
    return true;
}

std::vector<std::string> AscendDetectionEngine::getInputNames() const {
    return input_names_;
}

std::vector<std::string> AscendDetectionEngine::getOutputNames() const {
    return output_names_;
}

bool AscendDetectionEngine::loadModel(const DetectionInferenceConfig& config) {
    cleanup();
    config_ = config;
    device_id_ = config.device_id >= 0 ? config.device_id : 0;

    if (!initAcl()) {
        cleanup();
        return false;
    }

    aclError ret = aclmdlLoadFromFile(config.model_path.c_str(), &model_id_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendDetection] aclmdlLoadFromFile failed: {} ({})",
                      static_cast<int>(ret), config.model_path);
        cleanup();
        return false;
    }
    model_loaded_ = true;

    desc_ = aclmdlCreateDesc();
    if (!desc_ || aclmdlGetDesc(desc_, model_id_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendDetection] aclmdlGetDesc failed");
        cleanup();
        return false;
    }

    if (!parseIo()) {
        cleanup();
        return false;
    }

    LOG_INFO_FMT("[AscendDetection] Loaded '{}' on {} ({}), input {}x{}, {} outputs",
                 config.model_path, getBackendName(), device_id_,
                 config_.input_width, config_.input_height, outputs_.size());
    return true;
}

bool AscendDetectionEngine::parseIo() {
    const size_t in_count = aclmdlGetNumInputs(desc_);
    const size_t out_count = aclmdlGetNumOutputs(desc_);
    if (in_count == 0 || out_count == 0) {
        LOG_ERROR("[AscendDetection] Model has no input/output");
        return false;
    }

    // 输入：优先按名字 images 找，否则用第 0 个
    size_t input_index = 0;
    if (aclmdlGetInputIndexByName(desc_, "images", &input_index) != ACL_SUCCESS) {
        input_index = 0;
    }
    input_size_ = aclmdlGetInputSizeByIndex(desc_, input_index);
    aclDataType input_type = aclmdlGetInputDataType(desc_, input_index);
    const char* input_name = aclmdlGetInputNameByIndex(desc_, input_index);
    if (input_name) {
        input_name_ = input_name;
        input_names_.push_back(input_name_);
    }

    aclmdlIODims in_dims{};
    if (aclmdlGetInputDims(desc_, input_index, &in_dims) == ACL_SUCCESS) {
        if (in_dims.dimCount != 4 || in_dims.dims[0] != 1) {
            LOG_ERROR_FMT("[AscendDetection] Expect NCHW batch=1 input, got dimCount={}",
                          in_dims.dimCount);
            return false;
        }
        const int model_h = static_cast<int>(in_dims.dims[2]);
        const int model_w = static_cast<int>(in_dims.dims[3]);
        if (config_.input_width <= 0 || config_.input_height <= 0) {
            config_.input_width = model_w;
            config_.input_height = model_h;
        } else if (config_.input_width != model_w || config_.input_height != model_h) {
            LOG_ERROR_FMT("[AscendDetection] Model input is {}x{} but detector_config says {}x{}; "
                          "re-convert the .om or fix input_size",
                          model_w, model_h, config_.input_width, config_.input_height);
            return false;
        }
    }
    if (input_type != ACL_FLOAT) {
        LOG_ERROR_FMT("[AscendDetection] Unexpected input type {} (expect FP32)",
                      static_cast<int>(input_type));
        return false;
    }
    if (input_size_ == 0) {
        LOG_ERROR("[AscendDetection] Invalid input size");
        return false;
    }

    void* d_input = nullptr;
    if (aclrtMalloc(&d_input, input_size_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !d_input) {
        LOG_ERROR("[AscendDetection] aclrtMalloc input failed");
        return false;
    }
    d_input_ = d_input;
    input_data_buffer_ = aclCreateDataBuffer(d_input_, input_size_);
    input_ds_ = aclmdlCreateDataset();
    if (!input_data_buffer_ || !input_ds_ ||
        aclmdlAddDatasetBuffer(input_ds_, input_data_buffer_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendDetection] Create input dataset failed");
        return false;
    }

    // 输出：按名字匹配 5 张量契约
    outputs_.resize(out_count);
    output_ds_ = aclmdlCreateDataset();
    if (!output_ds_) {
        LOG_ERROR("[AscendDetection] Create output dataset failed");
        return false;
    }
    for (size_t i = 0; i < out_count; ++i) {
        auto& buf = outputs_[i];
        buf.size = aclmdlGetOutputSizeByIndex(desc_, i);
        buf.type = aclmdlGetOutputDataType(desc_, i);
        const char* name = aclmdlGetOutputNameByIndex(desc_, i);
        buf.name = name ? name : "";
        aclmdlIODims dims{};
        if (aclmdlGetOutputDims(desc_, i, &dims) == ACL_SUCCESS) {
            buf.elements = elementsFromDims(dims);
        }
        if (buf.size == 0) {
            LOG_ERROR_FMT("[AscendDetection] Output[{}] '{}' has zero size", i, buf.name);
            return false;
        }
        if (aclrtMalloc(&buf.dev, buf.size, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !buf.dev) {
            LOG_ERROR_FMT("[AscendDetection] aclrtMalloc output[{}] failed", i);
            return false;
        }
        buf.data_buffer = aclCreateDataBuffer(buf.dev, buf.size);
        if (!buf.data_buffer ||
            aclmdlAddDatasetBuffer(output_ds_, buf.data_buffer) != ACL_SUCCESS) {
            LOG_ERROR_FMT("[AscendDetection] Add output[{}] dataset buffer failed", i);
            return false;
        }
        output_names_.push_back(buf.name);
        for (const char* expected : kKnownOutputs) {
            if (buf.name.find(expected) != std::string::npos) {
                output_index_[expected] = i;
                break;
            }
        }
        LOG_INFO_FMT("[AscendDetection] output[{}] {} type={} size={} elems={}",
                     i, buf.name, static_cast<int>(buf.type), buf.size, buf.elements);
    }

    // 名字不完全匹配但正好 5 个输出：按标准顺序兜底
    if (output_index_.size() != 5 && out_count == 5) {
        LOG_WARN("[AscendDetection] Output names mismatch contract, using positional order");
        for (size_t i = 0; i < 5; ++i) output_index_[kKnownOutputs[i]] = i;
    }
    if (output_index_.size() != 5) {
        LOG_ERROR_FMT("[AscendDetection] Expected 5 detection outputs, got {} matched", output_index_.size());
        return false;
    }
    return true;
}

bool AscendDetectionEngine::setInputTensor(const std::string& name, void* gpu_ptr) {
    if (!name.empty()) input_name_ = name;
    input_ptr_ = gpu_ptr;
    return input_ptr_ != nullptr;
}

bool AscendDetectionEngine::setOutputTensor(const std::string& name, void* gpu_ptr) {
    external_outputs_[name] = gpu_ptr;
    return true;
}

bool AscendDetectionEngine::infer() {
    if (!model_loaded_ || !input_ptr_ || !input_ds_ || !output_ds_) {
        LOG_ERROR("[AscendDetection] infer called before model/input is ready");
        return false;
    }
    // ACL 的 device context 是线程相关的；worker 线程首次推理时补 setDevice
    aclError ret = aclrtSetDevice(device_id_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendDetection] aclrtSetDevice on worker thread failed: {}", static_cast<int>(ret));
        return false;
    }

    ret = aclrtMemcpy(d_input_, input_size_, input_ptr_, input_size_,
                      ACL_MEMCPY_HOST_TO_DEVICE);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendDetection] H2D input memcpy failed: {}", static_cast<int>(ret));
        return false;
    }

    ret = aclmdlExecute(model_id_, input_ds_, output_ds_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendDetection] aclmdlExecute failed: {}", static_cast<int>(ret));
        return false;
    }

    if (!copyOutputs()) return false;
    copyToExternalBuffers();
    return true;
}

namespace {

bool d2h(const void* src, void* dst, size_t bytes) {
    return aclrtMemcpy(dst, bytes, src, bytes, ACL_MEMCPY_DEVICE_TO_HOST) == ACL_SUCCESS;
}

} // namespace

bool AscendDetectionEngine::copyOutputs() {
    auto readFloats = [&](const char* name, size_t count, std::vector<float>& out) -> bool {
        auto it = output_index_.find(name);
        if (it == output_index_.end()) return false;
        const auto& b = outputs_[it->second];
        const size_t avail = b.elements ? b.elements : (b.size / typeSize(b.type));
        // count == 0 表示读取整个输出（用于从 scores 推导有效检测数）
        count = (count == 0) ? avail : std::min(count, avail);
        if (count == 0) {
            out.clear();
            return true;
        }
        out.resize(count);
        if (b.type == ACL_FLOAT) {
            return d2h(b.dev, out.data(), count * sizeof(float));
        }
        if (b.type == ACL_FLOAT16) {
            std::vector<uint16_t> h(count);
            if (!d2h(b.dev, h.data(), count * sizeof(uint16_t))) return false;
            for (size_t i = 0; i < count; ++i) out[i] = halfToFloat(h[i]);
            return true;
        }
        return false;
    };

    auto readI64s = [&](const char* name, size_t count, std::vector<int64_t>& out) -> bool {
        auto it = output_index_.find(name);
        if (it == output_index_.end()) return false;
        const auto& b = outputs_[it->second];
        const size_t avail = b.elements ? b.elements : (b.size / typeSize(b.type));
        count = (count == 0) ? avail : std::min(count, avail);
        if (count == 0) {
            out.clear();
            return true;
        }
        out.resize(count);
        if (b.type == ACL_INT64 || b.type == ACL_UINT64) {
            return d2h(b.dev, out.data(), count * sizeof(int64_t));
        }
        if (b.type == ACL_INT32 || b.type == ACL_UINT32) {
            std::vector<int32_t> v(count);
            if (!d2h(b.dev, v.data(), count * sizeof(int32_t))) return false;
            for (size_t i = 0; i < count; ++i) out[i] = v[i];
            return true;
        }
        if (b.type == ACL_FLOAT) {
            std::vector<float> v(count);
            if (!d2h(b.dev, v.data(), count * sizeof(float))) return false;
            for (size_t i = 0; i < count; ++i) out[i] = static_cast<int64_t>(v[i]);
            return true;
        }
        return false;
    };

    // 注意：ATC 转换后的 OM 里 det_num_dets 可能被固化成 max_det（实测恒为 200），
    // 不能作为有效检测数；且 det_scores/det_boxes 是 class-major 布局
    // （9 类 × 300 槽），有效槽位并不连续。这里读取完整输出后把 score>0 的槽位
    // 压缩到前 n 项，交给节点层统一后处理。
    std::vector<float> all_boxes, all_scores;
    std::vector<int64_t> all_classes, all_batch_ids;
    if (!readFloats(kBoxesName, 0, all_boxes) ||
        !readFloats(kScoresName, 0, all_scores) ||
        !readI64s(kClassesName, 0, all_classes) ||
        !readI64s(kBatchIdsName, 0, all_batch_ids)) {
        LOG_ERROR("[AscendDetection] Read output tensors failed");
        return false;
    }

    size_t count = all_scores.size();
    count = std::min(count, all_boxes.size() / 4);
    count = std::min(count, all_classes.size());
    count = std::min(count, all_batch_ids.size());

    std::vector<size_t> valid;
    valid.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        if (all_scores[i] > 0.0f) valid.push_back(i);
    }

    const size_t cap = config_.max_detections > 0
                           ? static_cast<size_t>(config_.max_detections)
                           : 200;
    if (valid.size() > cap) {
        std::partial_sort(valid.begin(), valid.begin() + cap, valid.end(),
                          [&](size_t a, size_t b) {
                              return all_scores[a] > all_scores[b];
                          });
        valid.resize(cap);
    }

    boxes_.clear();
    scores_.clear();
    classes_.clear();
    batch_ids_.clear();
    boxes_.reserve(valid.size() * 4);
    scores_.reserve(valid.size());
    classes_.reserve(valid.size());
    batch_ids_.reserve(valid.size());
    for (size_t i : valid) {
        boxes_.insert(boxes_.end(), all_boxes.begin() + i * 4, all_boxes.begin() + i * 4 + 4);
        scores_.push_back(all_scores[i]);
        classes_.push_back(all_classes[i]);
        batch_ids_.push_back(all_batch_ids[i]);
    }
    num_dets_ = static_cast<int64_t>(scores_.size());
    return true;
}

bool AscendDetectionEngine::copyToExternalBuffers() {
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

void* AscendDetectionEngine::getOutputTensor(const std::string& name) {
    if (name == kNumDetsName) return &num_dets_;
    if (name == kBoxesName) return boxes_.data();
    if (name == kScoresName) return scores_.data();
    if (name == kClassesName) return classes_.data();
    if (name == kBatchIdsName) return batch_ids_.data();
    return nullptr;
}

size_t AscendDetectionEngine::getOutputTensorSize(const std::string& name) const {
    if (name == kNumDetsName) return sizeof(int64_t);
    if (name == kBoxesName) return boxes_.size() * sizeof(float);
    if (name == kScoresName) return scores_.size() * sizeof(float);
    if (name == kClassesName) return classes_.size() * sizeof(int64_t);
    if (name == kBatchIdsName) return batch_ids_.size() * sizeof(int64_t);
    return 0;
}

void AscendDetectionEngine::cleanup() {
    if (input_ds_) {
        aclmdlDestroyDataset(input_ds_);
        input_ds_ = nullptr;
    }
    if (input_data_buffer_) {
        aclDestroyDataBuffer(input_data_buffer_);
        input_data_buffer_ = nullptr;
    }
    if (d_input_) {
        aclrtFree(d_input_);
        d_input_ = nullptr;
    }
    if (output_ds_) {
        aclmdlDestroyDataset(output_ds_);
        output_ds_ = nullptr;
    }
    for (auto& buf : outputs_) {
        if (buf.data_buffer) aclDestroyDataBuffer(buf.data_buffer);
        if (buf.dev) aclrtFree(buf.dev);
    }
    outputs_.clear();
    if (desc_) {
        aclmdlDestroyDesc(desc_);
        desc_ = nullptr;
    }
    if (model_loaded_) {
        aclmdlUnload(model_id_);
        model_loaded_ = false;
    }
    output_index_.clear();
    input_names_.clear();
    output_names_.clear();
    external_outputs_.clear();
    boxes_.clear();
    scores_.clear();
    classes_.clear();
    batch_ids_.clear();
    num_dets_ = 0;
    input_ptr_ = nullptr;
    input_size_ = 0;
    acl_device_set_ = false;
}

REGISTER_DETECTION_INFERENCE_BACKEND(DetectionBackend::ASCEND, AscendDetectionEngine)

} // namespace hal
} // namespace ai_stream
