// src/hal/infer/nvidia/tensorrt_relation.h
// TensorRT 视觉关系引擎（RelateAnything 关系图）
//
// 图的输入是「图像 + 区域」，不是纯图像：
//   image        [1, 3, S, S]   已在 letterbox 空间的图像
//   boxes        [1, N, 4]      xyxy 像素坐标，同一 letterbox 空间
//   box_counts   [1]             真实框数（图按固定 N 建，多的位置零填充）
//   W, alpha     [V, D], [V]    谓词向量——**图输入**，所以换谓词不用重导引擎
// 输出：
//   pred_logits  [1, K, V]      原始谓词 logit
//   pair_logits  [1, K]         原始配对存在性 logit
//   sub_idx/obj_idx [1, K]      主体/客体框下标
//   valid_mask   [1, K]         有效位
#pragma once

#include "ai_stream/hal/i_relation.h"
#include "../common/relation_decode.h"
#include "trt_core.h"

#include <memory>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

/**
 * @brief 谓词表：随模型分发，不随帧变
 */
struct PredicateBank {
    std::vector<std::string> names;          // 243 个候选谓词的全名表
    std::vector<float> thresholds;           // 逐谓词阈值（与 names 等长）
    std::vector<float> embed_W;              // [names.size() * dim]，行优先
    std::vector<float> embed_alpha;          // [names.size()]
    std::vector<std::string> default_predicates;  // 引擎自带的默认激活集
    float calib_a = 1.0f;
    float calib_b = 0.0f;
    int img_size = 448;
    int text_dim = 512;

    /**
     * 谓词嵌入维度，**必须从 W 和 alpha 的长度比推出来**，不能用 alpha 的长度：
     * alpha 是每个谓词一个标量（243 个），而嵌入维度是 text_dim（512）。
     * 两者混用会让 W 的行算错，读出完全错位的谓词向量，且不报错。
     */
    size_t dim() const
    {
        if (embed_alpha.empty() || embed_W.empty())
            return 0;
        return embed_W.size() / embed_alpha.size();
    }

    bool empty() const { return names.empty() || embed_alpha.empty() || embed_W.empty(); }
};

class TensorrtRelation : public IRelationEngine {
public:
    TensorrtRelation();
    ~TensorrtRelation() override;

    bool loadModel(const RelationConfig& config) override;
    bool infer(const RelationInput& input,
               std::vector<RelationTriplet>& out) override;
    bool inferPreprocessed(const float* input_nchw, size_t size_bytes,
                           const RelationInput& meta,
                           std::vector<RelationTriplet>& out) override;
    void unload() override;
    std::pair<int, int> getInputSize() const override;
    std::vector<std::string> getPredicates() const override;
    bool setPredicates(const std::vector<std::string>& predicates) override;
    float getScoreThreshold() const override;
    bool setScoreThreshold(float threshold) override;
    std::string getBackendName() const override { return "TensorRT Relation (NVIDIA)"; }
    bool isAvailable() const override;

private:
    /** 从 relateanything.json + predicate_bank.npz 导出的扁平 JSON 装载谓词表 */
    bool loadPredicateBank(const std::string& path, std::string& err);
    /** 把激活谓词的名字映射成 bank 行号，拼出 W/alpha 两个图输入 */
    bool buildPredicateInputs(std::string& err);
    /** image(BGR/RGB uint8, 已 letterbox) -> NCHW float 0..1，并写入 H2D */
    bool uploadImage(const RelationInput& input);
    bool runGraph();
    void copyOutputsToHost();
    void bindOutputAddresses();
    void allocOutput(const char* name, size_t bytes);
    /** 设置形状、搬框与谓词、推理并解码；infer 与 inferPreprocessed 共用 */
    bool runBoxes(const RelationInput& input, std::vector<RelationTriplet>& out);

    RelationConfig config_;
    PredicateBank bank_;
    RelationThresholdConfig thresholds_;
    bool loaded_ = false;

    TrtCore core_;
    std::vector<std::string> active_predicates_;

    // 引擎按固定 N 建，运行时形状与之对齐
    int64_t fixed_boxes_ = 32;
    int64_t fixed_predicates_ = 0;
    size_t pairs_ = 0;   // 引擎输出的候选配对数（不是框数）

    // host 侧暂存
    std::vector<float> h_pred_;
    std::vector<float> h_pair_;
    std::vector<int64_t> h_sub_;   // ONNX elem_type=int64，不能用 int32
    std::vector<int64_t> h_obj_;   // ONNX elem_type=int64
    std::vector<uint8_t> h_valid_;
    std::vector<float> h_w_;
    std::vector<float> h_alpha_;
    std::vector<float> box_scores_;
};

} // namespace hal
} // namespace ai_stream
