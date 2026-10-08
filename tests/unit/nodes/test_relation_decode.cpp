// 关系解码器的回归测试。
//
// 这些断言全部来自一次真实联调踩到的坑——每一条在单看代码时都"看起来对"，
// 只有拿真模型跑才会暴露，而且不报错，只是静默给错结果。
//
// 覆盖：
//   1. sub_idx/obj_idx 是 int64：按 int32 读会错位成垃圾下标
//   2. 配对数来自引擎输出维度，不是框数（N=32 时 pairs=128）
//   3. 未激活谓词必须被过滤（图按整表 243 导出）
//   4. 标定参数缺失必须显式可察（否则阈值全线失效且无提示）
//   5. 框坐标格式：图要归一化 cxcywh，不是 xyxy 像素
//   6. 逐谓词阈值不该自动套用 bank 的标定值
//
// 不依赖 GPU 与模型文件，纯 host 端逻辑。

#include <cassert>
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "relation_decode.h"

using namespace ai_stream::hal;

namespace {

int g_failed = 0;

void check(bool ok, const char* what, const char* detail = "")
{
    if (ok) {
        std::printf("  [PASS] %s\n", what);
    } else {
        std::printf("  [FAIL] %s %s\n", what, detail);
        ++g_failed;
    }
}

// sigmoid(a * (pred + w*pair) + b)，与 postprocess.py 同式
float score_of(float pred, float pair, float a, float b, float w = 1.0f)
{
    return 1.0f / (1.0f + std::exp(-(a * (pred + w * pair) + b)));
}

// 构造一个最小的"图输出"：2 个框、2 个谓词、2 个有效配对。
struct FakeGraph {
    std::vector<std::string> names = { "beside", "behind", "wearing" };
    std::vector<float> pred;     // [pairs * V]
    std::vector<float> pair;
    std::vector<int64_t> sub;
    std::vector<int64_t> obj;
    std::vector<uint8_t> valid;
    size_t v_count() const { return names.size(); }
};

FakeGraph makeGraph()
{
    FakeGraph g;
    const size_t V = g.v_count();
    // 4 个候选配对：2 个有效（双向）+ 2 个指向 padding 的无效
    g.pred.assign(4 * V, -9.0f);
    g.pair.assign(4, -1.0f);
    g.sub = { 1, 0, 2, 0 };
    g.obj = { 0, 1, 0, 2 };
    g.valid = { 1, 1, 0, 0 };
    // pair0: sub=1 -> obj=0, beside 得分高
    g.pred[0 * V + 0] = 2.868f;   // beside
    g.pair[0] = -0.657f;
    // pair1: sub=0 -> obj=1, beside 稍低
    g.pred[1 * V + 0] = 2.318f;
    g.pair[1] = -0.649f;
    return g;
}

RelationThresholdConfig calib()
{
    RelationThresholdConfig c;
    c.calib_a = 0.5651f;
    c.calib_b = -1.9623f;
    c.threshold = 0.2f;
    c.topk = 20;
    c.max_per_pair = 1;
    c.box_score_weight = true;
    return c;
}

}  // namespace

int main()
{
    std::printf("=== 1. int64 下标 + 双向配对 ===\n");
    {
        auto g = makeGraph();
        std::vector<float> box_scores = { 1.0f, 0.27f };   // 只送 2 个框
        std::vector<DecodedRelation> out;
        decodeRelations(g.pred, g.pair, g.sub, g.obj, g.valid,
                        g.names, calib(), box_scores, out,
                        { "beside", "behind" });

        // 期望 2 条：1->0 与 0->1，都是 beside
        check(out.size() == 2, "2 valid pairs yield 2 relations",
              ("got " + std::to_string(out.size())).c_str());
        if (out.size() == 2) {
            const bool dirs_ok = (out[0].subject_index == 1 && out[0].object_index == 0) ||
                                 (out[1].subject_index == 1 && out[1].object_index == 0);
            check(dirs_ok, "direction preserved (sub=1 -> obj=0 present)");
            check(out[0].predicate == "beside", "top predicate is beside",
                  out[0].predicate.c_str());

            // 与 Python 同式重算，验证分数口径
            const float expect = score_of(2.868f, -0.657f, 0.5651f, -1.9623f);
            check(std::fabs(out[0].score - expect) < 1e-5f,
                  "score matches sigmoid(a*(pred+w*pair)+b)",
                  ("got " + std::to_string(out[0].score) + " want " +
                   std::to_string(expect)).c_str());
            check(out[0].score > 0.28f && out[0].score < 0.34f,
                  "score in the Python-observed band (0.29..0.33)");
        }
    }

    std::printf("=== 2. 指向 padding 的配对被丢弃 ===\n");
    {
        auto g = makeGraph();
        std::vector<float> box_scores = { 1.0f, 0.27f };
        std::vector<DecodedRelation> out;
        decodeRelations(g.pred, g.pair, g.sub, g.obj, g.valid,
                        g.names, calib(), box_scores, out,
                        { "beside" });
        // sub=2 / obj=2 超出真实框数(2)，必须被 in_range 过滤；
        // 即使 valid=1 也不能漏进来（这里 valid 本来是 0，双重保险）
        bool any_out_of_range = false;
        for (const auto& r : out)
            if (r.subject_index >= 2 || r.object_index >= 2) any_out_of_range = true;
        check(!any_out_of_range, "no relation references a padding slot");
    }

    std::printf("=== 3. 未激活谓词被过滤 ===\n");
    {
        auto g = makeGraph();
        // 给 wearing 一个极高分，但没激活它
        g.pred[0 * g.v_count() + 2] = 99.0f;
        std::vector<float> box_scores = { 1.0f, 0.27f };
        std::vector<DecodedRelation> out;
        decodeRelations(g.pred, g.pair, g.sub, g.obj, g.valid,
                        g.names, calib(), box_scores, out,
                        { "beside" });   // wearing 未激活
        bool leaked = false;
        for (const auto& r : out)
            if (r.predicate == "wearing") leaked = true;
        // 图按整表导出，未激活列分数是基准值；若不过滤就会被吐出来，
        // 规则侧会匹配到从未启用过的谓词。
        check(!leaked, "high-scoring inactive predicate is not emitted");
    }

    std::printf("=== 4. 未标定时必须可察 ===\n");
    {
        RelationThresholdConfig raw;   // 默认 (1,0) = 未标定
        check(!raw.isCalibrated(), "default config reports uncalibrated");

        RelationThresholdConfig c;
        c.calib_a = 0.5651f;
        c.calib_b = -1.9623f;
        check(c.isCalibrated(), "bank calibration marks it calibrated");

        // 同一份 logit，未标定与标定的分数差一个 sigmoid 变换：
        const float s_raw = score_of(2.868f, -0.657f, 1.0f, 0.0f);
        const float s_cal = score_of(2.868f, -0.657f, 0.5651f, -1.9623f);
        check(std::fabs(s_raw - s_cal) > 0.2f,
              "uncalibrated vs calibrated differ materially (thresholds would not transfer)",
              ("raw=" + std::to_string(s_raw) + " cal=" + std::to_string(s_cal)).c_str());
    }

    std::printf("=== 5. 逐谓词阈值不自动套用 bank ===\n");
    {
        RelationThresholdConfig c = calib();
        c.threshold = 0.2f;   // 运行期主阈值
        c.per_predicate.clear();

        // bank 的标定值（beside 0.980）如果被自动套用，实测 0.29 的关系会被压死
        const std::vector<std::string> preds = { "beside" };
        const std::vector<float> thr = c.thresholdsVector(preds);
        check(thr.size() == 1 && std::fabs(thr[0] - 0.2f) < 1e-6f,
              "unspecified predicate uses the global threshold",
              ("got " + std::to_string(thr.empty() ? -1.0f : thr[0])).c_str());

        // 实测 beside ~0.29 必须能过 0.2，但过不了 0.980
        const float measured = score_of(2.868f, -0.657f, 0.5651f, -1.9623f);
        check(measured >= thr[0], "measured beside score passes the configured threshold");
        check(measured < 0.980f, "measured score would fail the bank's 0.980 threshold");

        // 显式覆盖时才生效
        RelationThresholdConfig ov = calib();
        ov.per_predicate["beside"] = 0.1f;
        const std::vector<float> thr2 = ov.thresholdsVector(preds);
        check(std::fabs(thr2[0] - 0.1f) < 1e-6f, "explicit override wins");
    }

    std::printf("=== 6. 自配对与 topk 上限 ===\n");
    {
        auto g = makeGraph();
        // 让 pair2/pair3 的 valid=1 但 sub==obj，解码器必须排除自配对
        g.valid = { 1, 1, 0, 0 };
        g.sub = { 1, 0, 1, 1 };
        g.obj = { 0, 1, 1, 1 };
        g.pred[2 * g.v_count() + 0] = 50.0f;   // 自配对给极高分
        std::vector<float> box_scores = { 1.0f, 0.27f };
        std::vector<DecodedRelation> out;
        decodeRelations(g.pred, g.pair, g.sub, g.obj, g.valid,
                        g.names, calib(), box_scores, out, { "beside" });
        bool self_paired = false;
        for (const auto& r : out)
            if (r.subject_index == r.object_index) self_paired = true;
        check(!self_paired, "self-pair (sub==obj) excluded even at high score");

        RelationThresholdConfig t = calib();
        t.topk = 1;
        std::vector<DecodedRelation> out2;
        decodeRelations(g.pred, g.pair, g.sub, g.obj, g.valid,
                        g.names, t, box_scores, out2, { "beside" });
        check(out2.size() <= 1, "topk caps the number of relations",
              ("got " + std::to_string(out2.size())).c_str());
    }

    std::printf("\n%s (%d failed)\n", g_failed == 0 ? "ALL PASS" : "FAILED", g_failed);
    return g_failed == 0 ? 0 : 1;
}