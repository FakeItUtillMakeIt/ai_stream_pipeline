// src/hal/infer/common/relation_decode.cpp
// RelateAnything 关系解码——对应 deploy/postprocess.py 的 decode()
#include "relation_decode.h"

#include <numeric>

namespace ai_stream {
namespace hal {

namespace {

/**
 * 关系分口径，**全链路唯一契约**，与 relsgg/scoring.py 的 ScoreContract 相同：
 *     score = sigmoid(a * (pred_logit + w * pair_logit) + b)
 *
 * 注意取的是 logits 而不是两次 sigmoid 后的概率：概率不可逆，而精度正是在
 * 那一步丢掉的。检测置信度刻意不进这个式子——conf(sub)*conf(obj) 通常只有
 * 0.1，会把所有分数拖到十分之一，让阈值变成一个和展示值无关的数字；它只参与
 * 排序折扣。
 */
inline float sigmoidf(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

} // namespace

void decodeRelations(const std::vector<float>& pred_logits,
                     const std::vector<float>& pair_logits,
                     const std::vector<int64_t>& sub_idx,
                     const std::vector<int64_t>& obj_idx,
                     const std::vector<uint8_t>& valid_mask,
                     const std::vector<std::string>& predicates,
                     const RelationThresholdConfig& cfg,
                     const std::vector<float>& box_scores,
                     std::vector<DecodedRelation>& out,
                     const std::vector<std::string>& only_predicates)
{
    out.clear();
    const size_t v_count = predicates.size();
    const size_t k_count = pair_logits.size();
    if (v_count == 0 || k_count == 0)
        return;
    if (pred_logits.size() < k_count * v_count)
        return;                       // 输出形状与契约不符，宁可不给结果

    const int64_t n_boxes = box_scores.empty()
                             ? -1
                             : static_cast<int64_t>(box_scores.size());

    // 图是按固定框数建的并做了零填充，所以无效槽位（以及指向 padding 的槽位）
    // 会带出 >= 真实框数的下标。必须在任何 gather 之前夹住，并且用 in_range 丢掉
    // ——不能只信 valid_mask，那不足以保证下标可寻址。
    std::vector<uint8_t> in_range(k_count, 1);
    std::vector<int64_t> sub(k_count), obj(k_count);
    for (size_t k = 0; k < k_count; ++k) {
        int64_t s = k < sub_idx.size() ? sub_idx[k] : -1;
        int64_t o = k < obj_idx.size() ? obj_idx[k] : -1;
        if (n_boxes >= 0) {
            const bool ok = (s >= 0 && o >= 0 && s < n_boxes && o < n_boxes);
            in_range[k] = ok ? 1u : 0u;
            s = std::clamp<int64_t>(s, 0, n_boxes - 1);
            o = std::clamp<int64_t>(o, 0, n_boxes - 1);
        } else {
            sub[k] = s;
            obj[k] = o;
            continue;
        }
        sub[k] = s;
        obj[k] = o;
    }

    // 关系分与排序分
    std::vector<float> score(k_count * v_count, 0.0f);
    std::vector<float> rank(k_count * v_count, 0.0f);
    std::vector<float> box_pair(k_count, 1.0f);
    if (cfg.box_score_weight && n_boxes > 0) {
        for (size_t k = 0; k < k_count; ++k) {
            const float bs = box_scores[static_cast<size_t>(sub[k])];
            const float bo = box_scores[static_cast<size_t>(obj[k])];
            box_pair[k] = bs * bo;
        }
    }
    for (size_t k = 0; k < k_count; ++k) {
        const float pair_logit = pair_logits[k];
        const float fused = cfg.pair_weight != 0.0f
                                ? pair_logit * cfg.pair_weight
                                : 0.0f;
        const float calib = cfg.calib_a * fused + cfg.calib_b;
        for (size_t v = 0; v < v_count; ++v) {
            const float s = sigmoidf(cfg.calib_a * pred_logits[k * v_count + v] + calib);
            score[k * v_count + v] = s;
            rank[k * v_count + v] = cfg.box_score_weight ? s * box_pair[k] : s;
        }
    }

    // 动态部分：逐谓词阈值向量 + valid_mask + 下标范围 + 自配对排除
    const std::vector<float> thr = cfg.thresholdsVector(predicates);
    std::vector<std::vector<uint8_t>> keep(k_count, std::vector<uint8_t>(v_count, 0));
    for (size_t k = 0; k < k_count; ++k) {
        const bool usable = in_range[k] != 0 && sub[k] != obj[k] &&
                           (k >= valid_mask.size() || valid_mask[k] != 0);
        if (!usable)
            continue;
        for (size_t v = 0; v < v_count; ++v) {
            keep[k][v] = score[k * v_count + v] >= thr[v] ? 1u : 0u;
            // 未激活的谓词直接剔掉：图按整表导出，这些列的 W/alpha 被填了 0，
            // 分数只是基准值，不代表任何启用过的关系。
            if (keep[k][v] && !only_predicates.empty() &&
                std::find(only_predicates.begin(), only_predicates.end(),
                          predicates[v]) == only_predicates.end())
                keep[k][v] = 0u;
        }
    }

    if (cfg.max_per_pair == 1) {
        // 每对框只留得分最高的那个谓词，避免同一关系重复上报
        for (size_t k = 0; k < k_count; ++k) {
            size_t best = v_count;
            for (size_t v = 0; v < v_count; ++v)
                if (keep[k][v] && (best == v_count || score[k * v_count + v] > score[k * v_count + best]))
                    best = v;
            for (size_t v = 0; v < v_count; ++v)
                if (v != best)
                    keep[k][v] = 0;
        }
    }

    struct Cand {
        size_t k;
        size_t v;
        float rank;
        float score;
    };
    std::vector<Cand> cands;
    cands.reserve(k_count);
    for (size_t k = 0; k < k_count; ++k) {
        bool any = false;
        for (size_t v = 0; v < v_count; ++v)
            any = any || keep[k][v] != 0;
        if (!any)
            continue;
        for (size_t v = 0; v < v_count; ++v)
            if (keep[k][v])
                cands.push_back({k, v, rank[k * v_count + v], score[k * v_count + v]});
    }
    if (cands.empty())
        return;

    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.rank > b.rank; });
    const size_t limit = cfg.topk > 0 ? static_cast<size_t>(cfg.topk) : cands.size();
    const size_t n = std::min(limit, cands.size());
    out.reserve(n);
    for (size_t i = 0; i < n; ++i) {
        const Cand& c = cands[i];
        DecodedRelation d;
        d.subject_index = sub[c.k];
        d.object_index = obj[c.k];
        d.predicate_index = static_cast<int>(c.v);
        d.predicate = predicates[c.v];
        d.score = c.score;
        d.rank = c.rank;
        out.push_back(std::move(d));
    }
}

} // namespace hal
} // namespace ai_stream
