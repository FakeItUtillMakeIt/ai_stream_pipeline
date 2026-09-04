// src/hal/rknn/rknn_detection_engine.h
// RKNN 检测推理引擎——Rockchip RK3588 NPU
//
// 输出契约与 TensorrtDetectionEngine 完全一致（5 张量语义）：
//   det_boxes     float  [total_dets * 4]  cx,cy,w,h（模型输入 640 坐标系）
//   det_scores    float  [total_dets]
//   det_classes   int64  [total_dets]
//   det_batch_ids int64  [total_dets]
//   det_num_dets  int64  [1]               批内总检测数
// 模型原始输出（无 NMS）由本引擎完成 decode + 按类别 NMS。
//
// 支持的模型输出形态（loadModel 时按 tensor 属性自动识别）：
//   V8_TRANS : 单输出 [1, 4+nc, N]（转置、已解码）
//   V8_DFL   : 单输出 [1, 64+nc, 8400]（三尺度 concat，DFL 解码）
//              或 3 分支输出 [1, 64+nc, h, w]（逐分支 DFL 解码）
//   V5       : 单输出 [1, N, 5+nc]
//
// 通过 dlopen 加载 librknnrt.so，与 rknn_inference_engine 保持一致，
// 允许在无 NPU 的编译主机构建（构建期仅需 rknn_api.h）。
#pragma once

#include "ai_stream/hal/i_detection_inference_engine.h"
#include "3rd_party/rk_platform/rknn/include/rknn_api.h"

#include <string>
#include <vector>
#include <map>
#include <mutex>

namespace ai_stream {
namespace hal {

class RknnDetectionEngine : public IDetectionInferenceEngine {
public:
    RknnDetectionEngine();
    ~RknnDetectionEngine() override;

    bool loadModel(const DetectionInferenceConfig& config) override;

    bool setInputTensor(const std::string& name, void* ptr) override;
    bool setOutputTensor(const std::string& name, void* ptr) override;
    void* getOutputTensor(const std::string& name) override;
    size_t getOutputTensorSize(const std::string& name) const override;
    bool allocateOutputBuffers() override;

    bool infer() override;
    bool inferAsync(void* stream) override;
    bool synchronize(void* stream) override;

    std::vector<std::string> getInputNames() const override;
    std::vector<std::string> getOutputNames() const override;
    std::pair<int, int> getInputSize() const override;
    int getMaxBatchSize() const override;
    std::string getBackendName() const override { return "RKNN Detection (Rockchip NPU)"; }
    bool isAvailable() const override;

    void* getRawContext() const override { return nullptr; }
    void* getRawEngine() const override { return nullptr; }

private:
    // 单帧推理：输入 host NCHW float [1,3,H,W]
    bool inferOne(const float* input_nchw, int batch_slot);

    enum class OutputLayout {
        UNKNOWN,
        V8_TRANS,   // 单输出 [1, 4+nc, N]
        V8_DFL,     // DFL 输出（单 concat 或 3 分支）
        V5          // 单输出 [1, N, 5+nc]
    };

    bool detectOutputLayout();

    // 候选收集：cx/cy/bw/bh（输入 640 坐标系）+ score + class
    bool collectCandidates(const std::vector<std::vector<float>>& outs,
                           std::vector<float>& cx, std::vector<float>& cy,
                           std::vector<float>& bw, std::vector<float>& bh,
                           std::vector<float>& score, std::vector<int>& cls);

    DetectionInferenceConfig config_;
    bool loaded_ = false;

    rknn_context rknn_ctx_ = 0;
    uint32_t n_inputs_ = 1;
    uint32_t n_outputs_ = 1;

    // 输入属性
    int input_width_ = 640;
    int input_height_ = 640;
    rknn_tensor_type input_type_ = RKNN_TENSOR_FLOAT32;
    rknn_tensor_format input_fmt_ = RKNN_TENSOR_NCHW;
    bool input_need_denorm_ = false;  // 模型输入 UINT8/INT8 时需反归一化 + 重排
    std::vector<float> mean_{0.0f, 0.0f, 0.0f};
    std::vector<float> std_{1.0f, 1.0f, 1.0f};

    // 输出布局
    OutputLayout layout_ = OutputLayout::UNKNOWN;
    int num_anchors_ = 0;
    int num_classes_ = 0;
    // V8_DFL 多分支时每分支的 (feat_w, stride)；单 concat 输出为空（内部按
    // 8400 = 80²+40²+20² 分段）
    struct Branch { int feat_w = 0; int stride = 0; };
    std::vector<Branch> branches_;

    // 引擎自管输出缓冲（5 张量，与 TRT 语义一致）
    std::vector<float> out_boxes_;
    std::vector<float> out_scores_;
    std::vector<int64_t> out_classes_;
    std::vector<int64_t> out_batch_ids_;
    int64_t out_num_dets_ = 0;

    // 外部设置的张量地址
    std::map<std::string, void*> tensor_ptrs_;

    std::string input_name_ = "images";
    std::string boxes_name_ = "det_boxes";
    std::string scores_name_ = "det_scores";
    std::string classes_name_ = "det_classes";
    std::string batch_ids_name_ = "det_batch_ids";
    std::string num_dets_name_ = "det_num_dets";

    int max_detections_ = 200;
    float nms_iou_ = 0.45f;

    // 反归一化暂存（UINT8 输入路径）
    std::vector<uint8_t> input_nhwc_;

    std::mutex mutex_;
};

} // namespace hal
} // namespace ai_stream
