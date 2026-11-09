// src/hal/infer/ascend/ascend_relation.h
// Ascend ACL RelateAnything 视觉关系引擎
//
// 图输入（ATC 固定 shape）：
//   image       [1,3,448,448] FP32
//   boxes       [1,32,4]      FP32 归一化 cxcywh
//   box_counts  [1]           INT64
//   W           [243,512]     FP32 谓词向量（整表）
//   alpha       [243]         FP32
// 输出：pred_logits / pair_logits / sub_idx / obj_idx / valid_mask
#pragma once

#include "ai_stream/hal/i_relation.h"
#include "infer/common/relation_decode.h"

#include <acl/acl.h>

#include <cstdint>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

struct RelPredicateBank {
    std::vector<std::string> names;
    std::vector<float> thresholds;
    std::vector<float> embed_W;
    std::vector<float> embed_alpha;
    std::vector<std::string> default_predicates;
    float calib_a = 1.0f;
    float calib_b = 0.0f;
    int img_size = 448;
    int text_dim = 512;
    size_t dim() const {
        if (embed_alpha.empty() || embed_W.empty()) return 0;
        return embed_W.size() / embed_alpha.size();
    }
    bool empty() const { return names.empty() || embed_W.empty() || embed_alpha.empty(); }
};

class AscendRelation : public IRelationEngine {
public:
    AscendRelation() = default;
    ~AscendRelation() override;

    bool loadModel(const RelationConfig& config) override;
    bool infer(const RelationInput& input,
               std::vector<RelationTriplet>& out) override;

    void unload() override;
    std::pair<int, int> getInputSize() const override { return {bank_.img_size, bank_.img_size}; }
    std::vector<std::string> getPredicates() const override { return active_predicates_; }
    bool setPredicates(const std::vector<std::string>& predicates) override;
    float getScoreThreshold() const override { return thresholds_.threshold; }
    bool setScoreThreshold(float threshold) override;
    std::string getBackendName() const override;
    bool isAvailable() const override;

private:
    bool initAcl();
    bool parseIo();
    bool loadPredicateBank(const std::string& path, std::string& err);
    bool buildPredicateInputs(std::string& err);
    bool uploadPredicateInputs();
    void cleanup();

    RelationConfig config_;
    RelPredicateBank bank_;
    RelationThresholdConfig thresholds_;
    std::string soc_name_;
    bool loaded_ = false;

    uint32_t model_id_ = 0;
    aclmdlDesc* desc_ = nullptr;
    aclmdlDataset* input_ds_ = nullptr;
    aclmdlDataset* output_ds_ = nullptr;

    std::vector<void*> d_in_;
    std::vector<size_t> in_size_;
    std::vector<aclDataType> in_type_;
    std::vector<aclDataBuffer*> in_buf_;
    std::vector<void*> d_out_;
    std::vector<size_t> out_size_;
    std::vector<aclDataType> out_type_;
    std::vector<aclDataBuffer*> out_buf_;

    int in_image_ = -1, in_boxes_ = -1, in_counts_ = -1, in_w_ = -1, in_alpha_ = -1;
    int out_pred_ = -1, out_pair_ = -1, out_sub_ = -1, out_obj_ = -1, out_valid_ = -1;

    int64_t fixed_boxes_ = 32;
    int64_t pairs_ = 0;
    int64_t fixed_predicates_ = 0;
    int img_size_ = 448;

    std::vector<std::string> active_predicates_;
    std::vector<float> h_w_;
    std::vector<float> h_alpha_;
    std::vector<float> h_pred_;
    std::vector<float> h_pair_;
    std::vector<int64_t> h_sub_;
    std::vector<int64_t> h_obj_;
    std::vector<uint8_t> h_valid_;
    std::vector<float> box_scores_;
};

} // namespace hal
} // namespace ai_stream
