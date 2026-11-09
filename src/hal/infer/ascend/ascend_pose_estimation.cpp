// src/hal/infer/ascend/ascend_pose_estimation.cpp
#include "ascend_pose_estimation.h"
#include "ai_stream/hal/pose_estimation_factory.h"
#include "ascend/ascend_acl.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <algorithm>
#include <cstring>
#include <unistd.h>
#include <vector>

namespace ai_stream {
namespace hal {
namespace {

constexpr size_t kCandidates = 8400;
constexpr size_t kPoseDim = 56;

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

AscendPoseEstimation::~AscendPoseEstimation() {
    cleanup();
}

bool AscendPoseEstimation::isAvailable() const {
    return ::access("/dev/davinci0", R_OK | W_OK) == 0;
}

std::string AscendPoseEstimation::getBackendName() const {
    return soc_name_.empty() ? "Ascend ACL Pose (Ascend NPU)"
                             : ("Ascend ACL Pose (" + soc_name_ + ")");
}

bool AscendPoseEstimation::initAcl() {
    if (!ensureAscendAclInitialized()) {
        LOG_ERROR("[AscendPose] aclInit failed");
        return false;
    }
    uint32_t count = 0;
    if (aclrtGetDeviceCount(&count) != ACL_SUCCESS || count == 0) {
        LOG_ERROR("[AscendPose] No Ascend device found");
        return false;
    }
    int device = config_.device_id >= 0 ? config_.device_id : 0;
    if (static_cast<uint32_t>(device) >= count) device = 0;
    if (aclrtSetDevice(device) != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendPose] aclrtSetDevice({}) failed", device);
        return false;
    }
    config_.device_id = device;
    const char* soc = aclrtGetSocName();
    if (soc) soc_name_ = soc;
    return true;
}

bool AscendPoseEstimation::loadModel(const PoseEstimationConfig& config) {
    cleanup();
    config_ = config;

    if (!initAcl()) {
        cleanup();
        return false;
    }

    aclError ret = aclmdlLoadFromFile(config.model_path.c_str(), &model_id_);
    if (ret != ACL_SUCCESS) {
        LOG_ERROR_FMT("[AscendPose] aclmdlLoadFromFile failed: {} ({}); only .om is supported",
                      static_cast<int>(ret), config.model_path);
        cleanup();
        return false;
    }
    loaded_ = true;

    desc_ = aclmdlCreateDesc();
    if (!desc_ || aclmdlGetDesc(desc_, model_id_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendPose] aclmdlGetDesc failed");
        cleanup();
        return false;
    }
    if (!parseIo()) {
        cleanup();
        return false;
    }

    LOG_INFO_FMT("[AscendPose] Loaded '{}' on {} (input={}x{}, output={} floats/person)",
                 config.model_path, getBackendName(), input_w_, input_h_, output_floats_);
    return true;
}

bool AscendPoseEstimation::parseIo() {
    const size_t in_count = aclmdlGetNumInputs(desc_);
    const size_t out_count = aclmdlGetNumOutputs(desc_);
    if (in_count != 1 || out_count == 0) {
        LOG_ERROR_FMT("[AscendPose] Expect 1 input, got {} / {} outputs", in_count, out_count);
        return false;
    }

    input_size_ = aclmdlGetInputSizeByIndex(desc_, 0);
    if (input_size_ == 0 || aclmdlGetInputDataType(desc_, 0) != ACL_FLOAT) {
        LOG_ERROR("[AscendPose] Input must be FP32");
        return false;
    }
    aclmdlIODims in_dims{};
    if (aclmdlGetInputDims(desc_, 0, &in_dims) == ACL_SUCCESS && in_dims.dimCount == 4) {
        input_h_ = static_cast<int>(in_dims.dims[2] > 0 ? in_dims.dims[2] : config_.input_height);
        input_w_ = static_cast<int>(in_dims.dims[3] > 0 ? in_dims.dims[3] : config_.input_width);
        if (in_dims.dims[0] != 1) {
            LOG_WARN_FMT("[AscendPose] Model batch={} (engine runs batch 1 per person)",
                         in_dims.dims[0]);
        }
    }
    const size_t per_person = static_cast<size_t>(input_w_) * input_h_ * 3 * sizeof(float);
    if (input_size_ < per_person) {
        LOG_ERROR_FMT("[AscendPose] Input size {} < per-person {}", input_size_, per_person);
        return false;
    }

    output_size_ = aclmdlGetOutputSizeByIndex(desc_, 0);
    if (output_size_ < kCandidates * kPoseDim * sizeof(float)) {
        LOG_ERROR_FMT("[AscendPose] Output too small: {} bytes", output_size_);
        return false;
    }
    aclmdlIODims out_dims{};
    if (aclmdlGetOutputDims(desc_, 0, &out_dims) == ACL_SUCCESS && out_dims.dimCount == 3) {
        const int64_t d1 = out_dims.dims[1];
        const int64_t d2 = out_dims.dims[2];
        if (d1 == static_cast<int64_t>(kCandidates) && d2 == static_cast<int64_t>(kPoseDim)) {
            output_needs_transpose_ = false;
        } else if (d1 == static_cast<int64_t>(kPoseDim) &&
                   d2 == static_cast<int64_t>(kCandidates)) {
            output_needs_transpose_ = true;
        } else {
            LOG_ERROR_FMT("[AscendPose] Unexpected output dims [1,{},{}]", d1, d2);
            return false;
        }
    } else {
        output_needs_transpose_ = false;
    }
    output_floats_ = kCandidates * kPoseDim;

    void* d_in = nullptr;
    if (aclrtMalloc(&d_in, input_size_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !d_in) {
        LOG_ERROR("[AscendPose] aclrtMalloc input failed");
        return false;
    }
    d_input_ = d_in;
    input_buf_ = aclCreateDataBuffer(d_input_, input_size_);
    input_ds_ = aclmdlCreateDataset();
    if (!input_buf_ || !input_ds_ ||
        aclmdlAddDatasetBuffer(input_ds_, input_buf_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendPose] create input dataset failed");
        return false;
    }

    output_ds_ = aclmdlCreateDataset();
    if (!output_ds_) {
        LOG_ERROR("[AscendPose] create output dataset failed");
        return false;
    }
    void* d_out = nullptr;
    if (aclrtMalloc(&d_out, output_size_, ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS || !d_out) {
        LOG_ERROR("[AscendPose] aclrtMalloc output failed");
        return false;
    }
    d_output_ = d_out;
    output_buf_ = aclCreateDataBuffer(d_output_, output_size_);
    if (!output_buf_ ||
        aclmdlAddDatasetBuffer(output_ds_, output_buf_) != ACL_SUCCESS) {
        LOG_ERROR("[AscendPose] add output dataset failed");
        return false;
    }
    return true;
}

bool AscendPoseEstimation::inferHost(const float* input_nchw, int num_persons,
                                     std::vector<float>& output_host) {
    if (!loaded_ || !input_nchw || num_persons <= 0) {
        LOG_ERROR("[AscendPose] inferHost before model ready");
        return false;
    }
    if (aclrtSetDevice(config_.device_id) != ACL_SUCCESS) {
        LOG_ERROR("[AscendPose] set device failed");
        return false;
    }

    const size_t per_person_input =
        static_cast<size_t>(input_w_) * input_h_ * 3 * sizeof(float);
    output_host.resize(static_cast<size_t>(num_persons) * output_floats_);

    std::vector<float> raw;
    if (output_needs_transpose_) raw.resize(output_floats_);

    for (int p = 0; p < num_persons; ++p) {
        const float* src = input_nchw + static_cast<size_t>(p) * input_w_ * input_h_ * 3;
        aclError ret = aclrtMemcpy(d_input_, input_size_, src, per_person_input,
                                   ACL_MEMCPY_HOST_TO_DEVICE);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR_FMT("[AscendPose] H2D person {} failed: {}", p, static_cast<int>(ret));
            return false;
        }
        ret = aclmdlExecute(model_id_, input_ds_, output_ds_);
        if (ret != ACL_SUCCESS) {
            LOG_ERROR_FMT("[AscendPose] aclmdlExecute person {} failed: {}",
                          p, static_cast<int>(ret));
            return false;
        }

        float* dst = output_host.data() + static_cast<size_t>(p) * output_floats_;
        if (!output_needs_transpose_) {
            ret = aclrtMemcpy(dst, output_floats_ * sizeof(float), d_output_,
                              output_floats_ * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
        } else {
            ret = aclrtMemcpy(raw.data(), output_floats_ * sizeof(float), d_output_,
                              output_floats_ * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
            if (ret == ACL_SUCCESS) {
                // [56, 8400] -> [8400, 56]
                for (size_t i = 0; i < kCandidates; ++i) {
                    for (size_t j = 0; j < kPoseDim; ++j) {
                        dst[i * kPoseDim + j] = raw[j * kCandidates + i];
                    }
                }
            }
        }
        if (ret != ACL_SUCCESS) {
            LOG_ERROR_FMT("[AscendPose] D2H person {} failed: {}", p, static_cast<int>(ret));
            return false;
        }
    }
    return true;
}

void AscendPoseEstimation::cleanup() {
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
    input_size_ = output_size_ = 0;
}

REGISTER_POSE_ESTIMATION_BACKEND(PoseEstimationBackend::ASCEND, AscendPoseEstimation)

} // namespace hal
} // namespace ai_stream
