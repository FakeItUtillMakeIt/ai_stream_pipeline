// src/hal/infer/horizon/horizon_detection_engine.cpp
// Horizon BPU 检测推理引擎实现——见头文件说明
#include "horizon_detection_engine.h"
#include "ai_stream/hal/detection_inference_engine_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <dlfcn.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <fstream>

extern "C" {
#include <hb_dnn.h>
#include <hb_ucp.h>
#include <hb_ucp_sys.h>
}

namespace ai_stream {
namespace hal {

// ---- dlopen 函数指针（与 horizon_inference_engine 相同模式）----
struct HorizonDnnApi {
    void* dl_handle = nullptr;
    int32_t (*InitializeFromFiles)(void**, char const**, int32_t) = nullptr;
    int32_t (*Release)(void*) = nullptr;
    int32_t (*GetModelNameList)(char const***, int32_t*, void*) = nullptr;
    int32_t (*GetModelHandle)(void**, void*, char const*) = nullptr;
    int32_t (*GetInputCount)(int32_t*, void*) = nullptr;
    int32_t (*GetOutputCount)(int32_t*, void*) = nullptr;
    int32_t (*GetInputTensorProperties)(struct hbDNNTensorProperties*, void*, int32_t) = nullptr;
    int32_t (*GetOutputTensorProperties)(struct hbDNNTensorProperties*, void*, int32_t) = nullptr;
    int32_t (*InferV2)(hbUCPTaskHandle_t*, struct hbDNNTensor*, struct hbDNNTensor const*, void*) = nullptr;
    int32_t (*WaitTaskDone)(hbUCPTaskHandle_t, int32_t) = nullptr;
    int32_t (*ReleaseTask)(hbUCPTaskHandle_t) = nullptr;
    int32_t (*Malloc)(struct hbUCPSysMem*, uint64_t, int32_t) = nullptr;
    int32_t (*MallocCached)(struct hbUCPSysMem*, uint64_t, int32_t) = nullptr;
    int32_t (*Free)(struct hbUCPSysMem*) = nullptr;
    int32_t (*MemFlush)(struct hbUCPSysMem const*, int32_t) = nullptr;
    int32_t (*SubmitTask)(hbUCPTaskHandle_t, struct hbUCPSchedParam const*) = nullptr;
    bool loaded = false;
};

static HorizonDnnApi& get_api() {
    static HorizonDnnApi api;
    static bool tried = false;
    if (tried) return api;
    tried = true;

    api.dl_handle = dlopen("libdnn.so", RTLD_NOW | RTLD_GLOBAL);
    if (!api.dl_handle) {
        api.dl_handle = dlopen("libhbrt4.so", RTLD_NOW | RTLD_GLOBAL);
    }
    if (!api.dl_handle) {
        LOG_WARN_FMT("[HorizonDetectionEngine] dlopen failed: {}", dlerror());
        return api;
    }

    auto sym = [&](const char* n) -> void* { return dlsym(api.dl_handle, n); };
    api.InitializeFromFiles = reinterpret_cast<decltype(api.InitializeFromFiles)>(sym("hbDNNInitializeFromFiles"));
    api.Release = reinterpret_cast<decltype(api.Release)>(sym("hbDNNRelease"));
    api.GetModelNameList = reinterpret_cast<decltype(api.GetModelNameList)>(sym("hbDNNGetModelNameList"));
    api.GetModelHandle = reinterpret_cast<decltype(api.GetModelHandle)>(sym("hbDNNGetModelHandle"));
    api.GetInputCount = reinterpret_cast<decltype(api.GetInputCount)>(sym("hbDNNGetInputCount"));
    api.GetOutputCount = reinterpret_cast<decltype(api.GetOutputCount)>(sym("hbDNNGetOutputCount"));
    api.GetInputTensorProperties = reinterpret_cast<decltype(api.GetInputTensorProperties)>(sym("hbDNNGetInputTensorProperties"));
    api.GetOutputTensorProperties = reinterpret_cast<decltype(api.GetOutputTensorProperties)>(sym("hbDNNGetOutputTensorProperties"));
    api.InferV2 = reinterpret_cast<decltype(api.InferV2)>(sym("hbDNNInferV2"));
    api.WaitTaskDone = reinterpret_cast<decltype(api.WaitTaskDone)>(sym("hbUCPWaitTaskDone"));
    api.ReleaseTask = reinterpret_cast<decltype(api.ReleaseTask)>(sym("hbUCPReleaseTask"));
    api.Malloc = reinterpret_cast<decltype(api.Malloc)>(sym("hbUCPMalloc"));
    api.MallocCached = reinterpret_cast<decltype(api.MallocCached)>(sym("hbUCPMallocCached"));
    api.Free = reinterpret_cast<decltype(api.Free)>(sym("hbUCPFree"));
    api.MemFlush = reinterpret_cast<decltype(api.MemFlush)>(sym("hbUCPMemFlush"));
    api.SubmitTask = reinterpret_cast<decltype(api.SubmitTask)>(sym("hbUCPSubmitTask"));

    api.loaded = api.InitializeFromFiles && api.Release && api.GetModelNameList &&
                 api.GetModelHandle && api.GetInputCount && api.GetOutputCount &&
                 api.GetInputTensorProperties && api.GetOutputTensorProperties &&
                 api.InferV2 && api.WaitTaskDone && api.ReleaseTask;
    if (!api.loaded) {
        LOG_ERROR("[HorizonDetectionEngine] Failed to load DNN API symbols");
    }
    return api;
}

// ---- NMS ----
static void nms(std::vector<float>& cx, std::vector<float>& cy,
                std::vector<float>& bw, std::vector<float>& bh,
                std::vector<float>& score, std::vector<int>& cls,
                float iou_thresh) {
    int n = static_cast<int>(cx.size());
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;

    std::sort(order.begin(), order.end(), [&](int a, int b) { return score[a] > score[b]; });

    std::vector<bool> suppressed(n, false);
    for (int i = 0; i < n; ++i) {
        if (suppressed[order[i]]) continue;
        int oi = order[i];
        for (int j = i + 1; j < n; ++j) {
            if (suppressed[order[j]]) continue;
            int oj = order[j];
            if (cls[oi] != cls[oj]) continue;

            float x1 = cx[oi] - bw[oi] / 2, y1 = cy[oi] - bh[oi] / 2;
            float x2 = cx[oi] + bw[oi] / 2, y2 = cy[oi] + bh[oi] / 2;
            float x3 = cx[oj] - bw[oj] / 2, y3 = cy[oj] - bh[oj] / 2;
            float x4 = cx[oj] + bw[oj] / 2, y4 = cy[oj] + bh[oj] / 2;

            float ix1 = std::max(x1, x3), iy1 = std::max(y1, y3);
            float ix2 = std::min(x2, x4), iy2 = std::min(y2, y4);
            float inter = std::max(0.f, ix2 - ix1) * std::max(0.f, iy2 - iy1);
            float area_i = bw[oi] * bh[oi];
            float area_j = bw[oj] * bh[oj];
            float iou = inter / (area_i + area_j - inter + 1e-6f);
            if (iou > iou_thresh) suppressed[oj] = true;
        }
    }

    std::vector<float> t_cx, t_cy, t_bw, t_bh, t_score;
    std::vector<int> t_cls;
    for (int i = 0; i < n; ++i) {
        if (!suppressed[order[i]]) {
            t_cx.push_back(cx[order[i]]);
            t_cy.push_back(cy[order[i]]);
            t_bw.push_back(bw[order[i]]);
            t_bh.push_back(bh[order[i]]);
            t_score.push_back(score[order[i]]);
            t_cls.push_back(cls[order[i]]);
        }
    }
    cx = std::move(t_cx);
    cy = std::move(t_cy);
    bw = std::move(t_bw);
    bh = std::move(t_bh);
    score = std::move(t_score);
    cls = std::move(t_cls);
}

HorizonDetectionEngine::HorizonDetectionEngine() {
    LOG_DEBUG("[HorizonDetectionEngine] Constructor");
}

HorizonDetectionEngine::~HorizonDetectionEngine() {
    if (dnn_packed_handle_) {
        auto& api = get_api();
        if (api.loaded && api.Release) {
            api.Release(dnn_packed_handle_);
        }
        dnn_packed_handle_ = nullptr;
    }
}

bool HorizonDetectionEngine::loadModel(const DetectionInferenceConfig& config) {
    std::lock_guard<std::mutex> lock(mutex_);
    config_ = config;
    max_detections_ = config.max_detections;
    LOG_INFO_FMT("[HorizonDetectionEngine] Loading model: {}", config.model_path);

    auto& api = get_api();
    if (!api.loaded) {
        LOG_ERROR("[HorizonDetectionEngine] DNN API not available");
        return false;
    }

    char const* model_files[] = {config.model_path.c_str()};
    int32_t ret = api.InitializeFromFiles(&dnn_packed_handle_, model_files, 1);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonDetectionEngine] InitializeFromFiles failed: {} (error: {})",
                       config.model_path, ret);
        return false;
    }

    char const** model_name_list = nullptr;
    int32_t model_count = 0;
    ret = api.GetModelNameList(&model_name_list, &model_count, dnn_packed_handle_);
    if (ret != 0 || model_count <= 0) {
        LOG_ERROR("[HorizonDetectionEngine] GetModelNameList failed");
        return false;
    }

    ret = api.GetModelHandle(&dnn_handle_, dnn_packed_handle_, model_name_list[0]);
    if (ret != 0) {
        LOG_ERROR("[HorizonDetectionEngine] GetModelHandle failed");
        return false;
    }

    // Query input
    ret = api.GetInputCount(&num_inputs_, dnn_handle_);
    if (ret != 0) return false;

    struct hbDNNTensorProperties props;
    ret = api.GetInputTensorProperties(&props, dnn_handle_, 0);
    if (ret != 0) return false;

    if (props.validShape.numDimensions >= 4) {
        input_width_ = props.validShape.dimensionSize[3];
        input_height_ = props.validShape.dimensionSize[2];
    } else {
        input_width_ = config.input_width;
        input_height_ = config.input_height;
    }
    LOG_INFO_FMT("[HorizonDetectionEngine] input: dims=[{} {} {} {}] type={} quantiType={} scaleLen={} scale0={} stride=[{} {} {} {}]",
                 props.validShape.dimensionSize[0], props.validShape.dimensionSize[1],
                 props.validShape.dimensionSize[2], props.validShape.dimensionSize[3],
                 props.tensorType, (int)props.quantiType, props.scale.scaleLen,
                 (props.scale.scaleData && props.scale.scaleLen > 0) ? props.scale.scaleData[0] : 0.0f,
                 props.stride[0], props.stride[1], props.stride[2], props.stride[3]);

    // Query outputs
    ret = api.GetOutputCount(&num_outputs_, dnn_handle_);
    if (ret != 0) return false;

    output_sizes_.resize(num_outputs_);
    for (int32_t i = 0; i < num_outputs_; ++i) {
        ret = api.GetOutputTensorProperties(&props, dnn_handle_, i);
        if (ret != 0) return false;
        output_sizes_[i] = props.alignedByteSize;
        std::string dims;
        for (int32_t d = 0; d < props.validShape.numDimensions; ++d)
            dims += std::to_string(props.validShape.dimensionSize[d]) + " ";
        LOG_INFO_FMT("[HorizonDetectionEngine] output[{}]: dims=[{}] alignedBytes={} type={} quantiType={} quantizeAxis={} scaleLen={} scale0={} stride=[{} {} {} {}]",
                     i, dims, props.alignedByteSize, props.tensorType, (int)props.quantiType,
                     props.quantizeAxis, props.scale.scaleLen,
                     (props.scale.scaleData && props.scale.scaleLen > 0) ? props.scale.scaleData[0] : 0.0f,
                     props.stride[0], props.stride[1], props.stride[2], props.stride[3]);

        // YOLOv8 单输出 [1, 4+nc, N]：直接从 shape 取类别数与 anchor 数，
        // 避免按整除猜测类别数导致输出错位（bugs: 14 类被误判为 80/20 类）。
        if (i == 0 && props.validShape.numDimensions == 3) {
            int64_t d1 = props.validShape.dimensionSize[1];
            int64_t d2 = props.validShape.dimensionSize[2];
            if (d1 > 4) { num_classes_ = static_cast<int>(d1 - 4); num_anchors_ = static_cast<int>(d2); }
            else if (d2 > 4) { num_classes_ = static_cast<int>(d2 - 4); num_anchors_ = static_cast<int>(d1); }
            // 行步长（元素个数）：BPU 输出行按对齐 padding，必须用 stride 索引
            if (props.stride[1] > 0) out0_row_floats_ = static_cast<int>(props.stride[1] / sizeof(float));
        }
    }

    loaded_ = true;
    LOG_INFO_FMT("[HorizonDetectionEngine] Model loaded: {}x{}, {} outputs",
                 input_width_, input_height_, num_outputs_);
    return true;
}

bool HorizonDetectionEngine::setInputTensor(const std::string& name, void* ptr) {
    tensor_ptrs_[name] = ptr;
    return true;
}

bool HorizonDetectionEngine::setOutputTensor(const std::string& name, void* ptr) {
    tensor_ptrs_[name] = ptr;
    return true;
}

void* HorizonDetectionEngine::getOutputTensor(const std::string& name) {
    auto it = tensor_ptrs_.find(name);
    if (it != tensor_ptrs_.end() && it->second) return it->second;

    // 返回引擎内部解码缓冲（infer() 后可用）
    if (name == boxes_name_ && !out_boxes_.empty()) return out_boxes_.data();
    if (name == scores_name_ && !out_scores_.empty()) return out_scores_.data();
    if (name == classes_name_ && !out_classes_.empty()) return out_classes_.data();
    if (name == batch_ids_name_ && !out_batch_ids_.empty()) return out_batch_ids_.data();
    if (name == num_dets_name_) return &out_num_dets_;
    return nullptr;
}

size_t HorizonDetectionEngine::getOutputTensorSize(const std::string& name) const {
    // Return estimated sizes for standard detection tensors
    int n = max_detections_;
    if (name == boxes_name_) return n * 4 * sizeof(float);
    if (name == scores_name_) return n * sizeof(float);
    if (name == classes_name_) return n * sizeof(int64_t);
    if (name == batch_ids_name_) return n * sizeof(int64_t);
    if (name == num_dets_name_) return sizeof(int64_t);
    return 0;
}

bool HorizonDetectionEngine::allocateOutputBuffers() {
    std::lock_guard<std::mutex> lock(mutex_);
    int n = max_detections_;
    out_boxes_.resize(n * 4);
    out_scores_.resize(n);
    out_classes_.resize(n);
    out_batch_ids_.resize(n);
    out_num_dets_ = 0;
    return true;
}

bool HorizonDetectionEngine::infer() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!loaded_ || !dnn_handle_) {
        LOG_ERROR("[HorizonDetectionEngine] Model not loaded");
        return false;
    }

    auto& api = get_api();
    if (!api.loaded) return false;

    // Find input tensor
    void* input_ptr = nullptr;
    for (auto& [name, ptr] : tensor_ptrs_) {
        if (name == input_name_ || name == "images" || name == "input") {
            input_ptr = ptr;
            break;
        }
    }
    if (!input_ptr) {
        LOG_ERROR("[HorizonDetectionEngine] Input tensor not set");
        return false;
    }

    // Prepare input tensor
    struct hbDNNTensorProperties input_props;
    api.GetInputTensorProperties(&input_props, dnn_handle_, 0);

    // 获取输入形状: [1, 640, 640, 3] NHWC
    int32_t batch = input_props.validShape.dimensionSize[0];
    int32_t h = input_props.validShape.dimensionSize[1];
    int32_t w = input_props.validShape.dimensionSize[2];
    int32_t c = input_props.validShape.dimensionSize[3];
    // tensorType: 2=S8, 3=U8, 7=F32
    bool is_int8 = (input_props.tensorType == 2 || input_props.tensorType == 3);

    struct hbUCPSysMem input_mem;
    size_t input_size = static_cast<size_t>(input_props.alignedByteSize);
    // 使用 MallocCached（与官方 hrt_model_exec / 样例一致，配合 MemFlush）
    int32_t ret = api.MallocCached(&input_mem, input_size, 0);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonDetectionEngine] Input MallocCached failed: {}", ret);
        return false;
    }
    // 实际数据大小（不含对齐填充）
    size_t data_size = batch * h * w * c * (is_int8 ? 1 : 4);

    if (is_int8) {
        // 预处理节点输出为归一化 float32 NCHW [0,1]；模型期望 [0,255] 量化为
        // int8 NHWC [-128,127]。先还原到 [0,255] 再平移。
        const float* src = static_cast<const float*>(input_ptr);
        int8_t* dst = static_cast<int8_t*>(input_mem.virAddr);

        // stride 信息: [batch_stride, h_stride, w_stride, c_stride]
        int64_t batch_stride = input_props.stride[0];
        int64_t h_stride = input_props.stride[1];
        int64_t w_stride = input_props.stride[2];

        for (int64_t i = 0; i < batch; ++i) {
            for (int hi = 0; hi < h; ++hi) {
                for (int wi = 0; wi < w; ++wi) {
                    for (int ci = 0; ci < c; ++ci) {
                        int64_t nchw_idx = ((i * c + ci) * h + hi) * w + wi;
                        float val = src[nchw_idx];
                        // [0,1] -> [0,255] -> [-128,127]
                        int32_t q = static_cast<int32_t>(std::round(val * 255.0f - 128.0f));
                        int8_t qval = static_cast<int8_t>(std::max(-128, std::min(127, q)));
                        // 按 stride 定位目标位置
                        int64_t dst_offset = i * batch_stride + hi * h_stride + wi * w_stride + ci;
                        dst[dst_offset] = qval;
                    }
                }
            }
        }
    } else {
        // float32 NCHW -> float32 NHWC，按 stride 拷贝
        const float* src = static_cast<const float*>(input_ptr);
        float* dst = static_cast<float*>(input_mem.virAddr);

        int64_t batch_stride = input_props.stride[0];
        int64_t h_stride = input_props.stride[1];
        int64_t w_stride = input_props.stride[2];

        for (int64_t i = 0; i < batch; ++i) {
            for (int hi = 0; hi < h; ++hi) {
                for (int wi = 0; wi < w; ++wi) {
                    for (int ci = 0; ci < c; ++ci) {
                        int64_t nchw_idx = ((i * c + ci) * h + hi) * w + wi;
                        int64_t dst_offset = (i * batch_stride + hi * h_stride + wi * w_stride + ci * sizeof(float)) / sizeof(float);
                        dst[dst_offset] = src[nchw_idx];
                    }
                }
            }
        }
    }

    struct hbDNNTensor input_tensor;
    input_tensor.sysMem = input_mem;
    input_tensor.properties = input_props;

    // 刷新输入缓存: HB_SYS_MEM_CACHE_CLEAN=2 (CPU→设备)，把 CPU 写入刷到设备可见内存
    api.MemFlush(&input_mem, 2);

    LOG_DEBUG_FMT("[HorizonDetectionEngine] Input: tensorType={}, quantiType={}, alignedByteSize={}, shape=[{},{},{},{}]",
        input_props.tensorType, input_props.quantiType, (long)input_props.alignedByteSize,
        input_props.validShape.dimensionSize[0], input_props.validShape.dimensionSize[1],
        input_props.validShape.dimensionSize[2], input_props.validShape.dimensionSize[3]);

    // Prepare output tensors
    std::vector<struct hbDNNTensor> output_tensors(num_outputs_);
    std::vector<struct hbUCPSysMem> output_mems(num_outputs_);

    for (int32_t i = 0; i < num_outputs_; ++i) {
        ret = api.MallocCached(&output_mems[i], output_sizes_[i], 0);
        if (ret != 0) {
            LOG_ERROR_FMT("[HorizonDetectionEngine] Output Malloc({}) failed: {}", i, ret);
            api.Free(&input_mem);
            for (int32_t j = 0; j < i; ++j) api.Free(&output_mems[j]);
            return false;
        }
        memset(output_mems[i].virAddr, 0, output_sizes_[i]);

        struct hbDNNTensorProperties out_props;
        api.GetOutputTensorProperties(&out_props, dnn_handle_, i);

        output_tensors[i].sysMem = output_mems[i];
        output_tensors[i].properties = out_props;
    }

    // Run inference
    hbUCPTaskHandle_t task_handle = nullptr;
    ret = api.InferV2(&task_handle, output_tensors.data(), &input_tensor, dnn_handle_);
    LOG_DEBUG_FMT("[HorizonDetectionEngine] InferV2 ret={}, task_handle={}", ret, (void*)task_handle);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonDetectionEngine] InferV2 failed: {}", ret);
        api.Free(&input_mem);
        for (auto& m : output_mems) api.Free(&m);
        return false;
    }

    // 提交任务到 BPU（必须在 WaitTaskDone 之前调用）
    hbUCPSchedParam sched_param;
    HB_UCP_INITIALIZE_SCHED_PARAM(&sched_param);
    sched_param.priority = 0;
    sched_param.backend = HB_UCP_BPU_CORE_ANY;
    ret = api.SubmitTask(task_handle, &sched_param);
    LOG_DEBUG_FMT("[HorizonDetectionEngine] SubmitTask ret={}", ret);
    if (ret != 0) {
        LOG_ERROR_FMT("[HorizonDetectionEngine] SubmitTask failed: {}", ret);
        api.ReleaseTask(task_handle);
        api.Free(&input_mem);
        for (auto& m : output_mems) api.Free(&m);
        return false;
    }

    ret = api.WaitTaskDone(task_handle, 5000);
    LOG_DEBUG_FMT("[HorizonDetectionEngine] WaitTaskDone ret={}", ret);
    if (ret != 0) {
        LOG_ERROR("[HorizonDetectionEngine] WaitTaskDone failed");
        api.ReleaseTask(task_handle);
        api.Free(&input_mem);
        for (auto& m : output_mems) api.Free(&m);
        return false;
    }

    // 刷新输出缓存: INVALIDATE = 1 (设备→CPU)
    for (int32_t i = 0; i < num_outputs_; ++i) {
        api.MemFlush(&output_mems[i], 1);
    }

    // Decode output (YOLOv8 style: single output [1, 4+nc, N])
    if (num_outputs_ >= 1) {
        const float* raw_output = static_cast<const float*>(output_mems[0].virAddr);
        int64_t raw_size = output_sizes_[0] / sizeof(float);

        std::vector<float> cx, cy, bw, bh, score;
        std::vector<int> cls;
        decodeYolov8(raw_output, raw_size, cx, cy, bw, bh, score, cls);

        // Apply NMS
        nms(cx, cy, bw, bh, score, cls, nms_iou_);

        // Fill output tensors
        out_num_dets_ = std::min(static_cast<int64_t>(cx.size()),
                                  static_cast<int64_t>(max_detections_));

        out_boxes_.resize(out_num_dets_ * 4);
        out_scores_.resize(out_num_dets_);
        out_classes_.resize(out_num_dets_);
        out_batch_ids_.resize(out_num_dets_);

        for (int64_t i = 0; i < out_num_dets_; ++i) {
            out_boxes_[i * 4 + 0] = cx[i];
            out_boxes_[i * 4 + 1] = cy[i];
            out_boxes_[i * 4 + 2] = bw[i];
            out_boxes_[i * 4 + 3] = bh[i];
            out_scores_[i] = score[i];
            out_classes_[i] = cls[i];
            out_batch_ids_[i] = 0;
        }
    }

    // Copy to external tensor pointers if set
    if (auto it = tensor_ptrs_.find(boxes_name_); it != tensor_ptrs_.end() && it->second) {
        memcpy(it->second, out_boxes_.data(), out_boxes_.size() * sizeof(float));
    }
    if (auto it = tensor_ptrs_.find(scores_name_); it != tensor_ptrs_.end() && it->second) {
        memcpy(it->second, out_scores_.data(), out_scores_.size() * sizeof(float));
    }
    if (auto it = tensor_ptrs_.find(classes_name_); it != tensor_ptrs_.end() && it->second) {
        memcpy(it->second, out_classes_.data(), out_classes_.size() * sizeof(int64_t));
    }
    if (auto it = tensor_ptrs_.find(batch_ids_name_); it != tensor_ptrs_.end() && it->second) {
        memcpy(it->second, out_batch_ids_.data(), out_batch_ids_.size() * sizeof(int64_t));
    }
    if (auto it = tensor_ptrs_.find(num_dets_name_); it != tensor_ptrs_.end() && it->second) {
        *static_cast<int64_t*>(it->second) = out_num_dets_;
    }

    api.ReleaseTask(task_handle);
    api.Free(&input_mem);
    for (auto& m : output_mems) api.Free(&m);

    return true;
}

bool HorizonDetectionEngine::inferAsync(void* stream) {
    (void)stream;
    // Horizon BPU doesn't support async inference via stream; fall back to sync
    return infer();
}

bool HorizonDetectionEngine::synchronize(void* stream) {
    (void)stream;
    return true;
}

bool HorizonDetectionEngine::decodeYolov8(const float* output, int output_size,
                                           std::vector<float>& cx_out, std::vector<float>& cy_out,
                                           std::vector<float>& bw_out, std::vector<float>& bh_out,
                                           std::vector<float>& score_out, std::vector<int>& cls_out) {
    // YOLOv8 output: [1, 4+nc, N] (transposed)
    // First 4 rows are cx, cy, bw, bh; remaining rows are class scores
    if (output_size < 4) return false;

    // Determine dimensions
    // output_size = (4 + nc) * N
    // We need to figure out nc (number of classes)
    // For simplicity, assume common YOLOv8 formats
    int nc = 0;
    int N = 0;

    // 优先使用模型输出 shape 提供的类别/anchor 数（可靠）
    if (num_classes_ > 0 && num_anchors_ > 0) {
        nc = num_classes_;
        N = num_anchors_;
    } else {
        // 回退：按整除猜测类别数
        for (int try_nc : {80, 20, 1, 2, 5, 10, 15, 30, 40, 60}) {
            if (output_size % (4 + try_nc) == 0) {
                nc = try_nc;
                N = output_size / (4 + nc);
                break;
            }
        }
    }

    if (nc == 0 || N == 0) {
        LOG_WARN_FMT("[HorizonDetectionEngine] Cannot determine output shape: size={}", output_size);
        return false;
    }

    // 行步长（元素个数）：优先用模型 stride（含对齐 padding），否则退回 N
    const int row = (out0_row_floats_ > 0) ? out0_row_floats_ : N;

    cx_out.clear();
    cy_out.clear();
    bw_out.clear();
    bh_out.clear();
    score_out.clear();
    cls_out.clear();

    for (int i = 0; i < N; ++i) {
        float cx = output[0 * row + i];
        float cy = output[1 * row + i];
        float bw = output[2 * row + i];
        float bh = output[3 * row + i];

        // Find best class
        float max_score = 0;
        int max_cls = 0;
        for (int c = 0; c < nc; ++c) {
            float s = output[(4 + c) * row + i];
            if (s > max_score) {
                max_score = s;
                max_cls = c;
            }
        }

        // Confidence threshold
        if (max_score < 0.25f) continue;

        cx_out.push_back(cx);
        cy_out.push_back(cy);
        bw_out.push_back(bw);
        bh_out.push_back(bh);
        score_out.push_back(max_score);
        cls_out.push_back(max_cls);
    }

    return true;
}

std::vector<std::string> HorizonDetectionEngine::getInputNames() const {
    return {input_name_};
}

std::vector<std::string> HorizonDetectionEngine::getOutputNames() const {
    return {boxes_name_, scores_name_, classes_name_, batch_ids_name_, num_dets_name_};
}

std::pair<int, int> HorizonDetectionEngine::getInputSize() const {
    return {input_width_, input_height_};
}

int HorizonDetectionEngine::getMaxBatchSize() const {
    return config_.max_batch_size;
}

bool HorizonDetectionEngine::isAvailable() const {
    return get_api().loaded;
}

REGISTER_DETECTION_INFERENCE_BACKEND(DetectionBackend::HORIZON, HorizonDetectionEngine)

} // namespace hal
} // namespace ai_stream
