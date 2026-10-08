// src/hal/infer/common/relation_decode.h
// RelateAnything 关系图输出的 host 端解码。
//
// 这一段逐行对应 RelateAnything/deploy/postprocess.py 的 decode()，改动时两边
// 必须同步——分数口径不一致会导致"离线评测说达标、线上阈值全错"，而且不报错。
//
// 上游图只吐原始 logits（pred_logits / pair_logits / sub_idx / obj_idx /
// valid_mask），阈值、逐谓词阈值、配对权重全部留在 host，这样每帧可调而不用
// 重新导出引擎。
#pragma once

#include <algorithm>
#include <cmath>
#include <string>
#include <unordered_map>
#include <vector>

namespace ai_stream {
namespace hal {

/**
 * @brief 一条解码后的三元组
 */
struct DecodedRelation {
    int subject_index = -1;
    int object_index = -1;
    int predicate_index = -1;
    std::string predicate;
    float score = 0.0f;      // 关系分，被阈值比较、也是对外展示的那个数
    float rank = 0.0f;       // 排序分，可按框置信度折扣
};

/**
 * @brief 阈值与标定配置，全部运行期可变
 */
struct RelationThresholdConfig {
    float threshold = 0.40f;                        // 全局关系分门限
    std::unordered_map<std::string, float> per_predicate;  // 逐谓词覆盖
    float pair_weight = 1.0f;                       // 配对存在性 logit 的权重
    float calib_a = 1.0f;                           // 标定；(1,0) 表示未标定
    float calib_b = 0.0f;
    int topk = 20;
    int max_per_pair = 1;                           // 一对框最多出几个谓词
    bool box_score_weight = true;                   // 排序时乘框置信度

    bool isCalibrated() const
    {
        return calib_a != 1.0f || calib_b != 0.0f;
    }

    /**
     * 逐谓词阈值向量：未指定的一律用全局值。
     */
    std::vector<float> thresholdsVector(const std::vector<std::string>& predicates) const
    {
        std::vector<float> out(predicates.size(), threshold);
        for (size_t i = 0; i < predicates.size(); ++i) {
            auto it = per_predicate.find(predicates[i]);
            if (it != per_predicate.end())
                out[i] = it->second;
        }
        return out;
    }
};

/**
 * @brief 解码原始图输出
 *
 * @param pred_logits    [K][V] 原始谓词 logit
 * @param pair_logits    [K]    原始配对存在性 logit
 * @param sub_idx        [K]    主体框下标
 * @param obj_idx        [K]    客体框下标
 * @param valid_mask     [K]    有效位
 * @param predicates     谓词名表，长度必须等于 V
 * @param cfg            阈值配置
 * @param box_scores     可选，框置信度；给了就参与排序折扣
 * @param out            已按 rank 降序、长度不超过 cfg.topk
 */
void decodeRelations(const std::vector<float>& pred_logits,
                     const std::vector<float>& pair_logits,
                     const std::vector<int64_t>& sub_idx,   // ONNX elem_type=int64
                     const std::vector<int64_t>& obj_idx,
                     const std::vector<uint8_t>& valid_mask,
                     const std::vector<std::string>& predicates,
                     const RelationThresholdConfig& cfg,
                     const std::vector<float>& box_scores,
                     std::vector<DecodedRelation>& out,
                     /**
                      * 只输出这些谓词。空表示不过滤。
                      *
                      * 图按**整张谓词表**导出（num_predicates 是静态维），所以
                      * pred_logits 永远有全部列；实际激活的只是一个子集。
                      * 没有这个过滤，解码会把未激活谓词（那些行 W/alpha 被填 0）
                      * 也当作结果吐出来，规则侧就会匹配到没启用过的谓词。
                      */
                     const std::vector<std::string>& only_predicates = {});

} // namespace hal
} // namespace ai_stream
