// src/hal/infer/horizon/horizon_detection_engine.h
// Horizon BPU 检测推理引擎——地平线 RDK S100P
//
// 输出契约与 TensorRT/RKNN 一致（5 张量语义）：
//   det_boxes     float  [total_dets * 4]  cx,cy,w,h（模型输入 640 坐标系）
//   det_scores    float  [total_dets]
//   det_classes   int64  [total_dets]
//   det_batch_ids int64  [total_dets]
//   det_num_dets  int64  [1]               批内总检测数
//
// 通过 dlopen 加载 libdnn.so，允许在无 BPU 的编译主机构建。
#pragma once

#include "ai_stream/hal/i_detection_inference_engine.h"
#include "ai_stream/hal/i_detection_capabilities.h"
#include <string>
#include <vector>
#include <map>
#include <mutex>

namespace ai_stream {
namespace hal {

class HorizonDetectionEngine : public IDetectionInferenceEngine, public INv12Input {
public:
    HorizonDetectionEngine();
    ~HorizonDetectionEngine() override;

    bool loadModel(const DetectionInferenceConfig& config) override;

    bool setInputTensor(const std::string& name, void* ptr) override;
    // NV12 输入直通（NV12 输入模型）：设置紧凑 NV12，infer() 将走 NV12 分支
    bool setNv12Input(const uint8_t* nv12, int width, int height) override;
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
    std::string getBackendName() const override { return "Horizon BPU Detection (RDK)"; }
    bool isAvailable() const override;

private:
    bool loadDnnLib();
    bool detectOutputLayout();

    // YOLO-style decode + NMS
    bool decodeYolov8(const float* output, int output_size,
                      std::vector<float>& cx_out, std::vector<float>& cy_out,
                      std::vector<float>& bw_out, std::vector<float>& bh_out,
                      std::vector<float>& score_out, std::vector<int>& cls_out);

    DetectionInferenceConfig config_;
    bool loaded_ = false;

    // Horizon DNN handles (opaque pointers)
    void* dnn_packed_handle_ = nullptr;
    void* dnn_handle_ = nullptr;

    int input_width_ = 640;
    int input_height_ = 640;
    int num_inputs_ = 1;
    int num_outputs_ = 1;
    std::vector<int64_t> output_sizes_;

    // NV12 输入路径
    bool use_nv12_ = false;          // 模型是否为 NV12 双输入（Y/UV）
    const uint8_t* nv12_ptr_ = nullptr;  // 当前帧紧凑 NV12
    int nv12_w_ = 0;
    int nv12_h_ = 0;

    // Output layout
    int num_anchors_ = 0;
    int num_classes_ = 0;
    int out0_row_floats_ = 0;   // 输出张量每行真实步长（float 数，含对齐 padding）

    // Engine-managed output buffers (5 tensors)
    std::vector<float> out_boxes_;
    std::vector<float> out_scores_;
    std::vector<int64_t> out_classes_;
    std::vector<int64_t> out_batch_ids_;
    int64_t out_num_dets_ = 0;

    // External tensor pointers
    std::map<std::string, void*> tensor_ptrs_;

    std::string input_name_ = "images";
    std::string boxes_name_ = "det_boxes";
    std::string scores_name_ = "det_scores";
    std::string classes_name_ = "det_classes";
    std::string batch_ids_name_ = "det_batch_ids";
    std::string num_dets_name_ = "det_num_dets";

    int max_detections_ = 200;
    float nms_iou_ = 0.45f;

    // dlopen handle
    void* dl_handle_ = nullptr;

    std::mutex mutex_;
};

} // namespace hal
} // namespace ai_stream
