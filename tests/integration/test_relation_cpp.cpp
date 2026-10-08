// C++ 侧关系链路的真实推理测试。
//
// 用真实的 TensorRT engine + 真实谓词库 + 真实图片，跑通
//   loadModel -> setPredicates -> infer
// 用来把"模型/后端本身的问题"和"管道集成的问题"分开：这里不碰管道配置、
// 帧调度和规则，只验后端。
//
// 结果要与 RelateAnything/deploy/runtime.py 的 Python 输出一致；
// tools/compare_relation.py 会把两边结果拉出来逐条比。
//
// 编译（ai_stream_pipeline 下）：
//   g++ -std=c++17 -O2 -Iinclude -Isrc -Isrc/hal/infer/common -I3rd_party \
//       tools/test_relation_cpp.cpp build/src/hal/libhal.a ... -o /tmp/test_relation

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "ai_stream/hal/relation_factory.h"
#include "3rd_party/log_mgr/log_mgr.h"

using namespace ai_stream;

namespace {

// 与 deploy/runtime.py:126-128 一致：letterbox(黑边) + BGR->RGB + /255。
// 注意本测试只做 letterbox，归一化在后端内部做。
// 逐行对应 deploy/runtime.py:36-50 的 letterbox()。
// 两处容易写错、且会导致框映射错位：
//   1. 缩放取 r = min(new/h, new/w)，即**长边**贴到 new。写成 max 会让
//      宽图算出负 padding（700x393 取 max 时 pad_x=-175）。
//   2. 填充色是灰 114（ultralytics 约定），不是黑 0。
cv::Mat letterbox(const cv::Mat& src, int size, float& scale, float& pad_x, float& pad_y)
{
    const float r = std::min(static_cast<float>(size) / src.rows,
                             static_cast<float>(size) / src.cols);
    const int nh = static_cast<int>(std::round(src.rows * r));
    const int nw = static_cast<int>(std::round(src.cols * r));
    pad_x = static_cast<float>((size - nw) / 2);
    pad_y = static_cast<float>((size - nh) / 2);

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(nw, nh), 0, 0, cv::INTER_LINEAR);
    cv::Mat canvas(size, size, CV_8UC3, cv::Scalar(114, 114, 114));
    resized.copyTo(canvas(cv::Rect(cv::Point(static_cast<int>(pad_x),
                                             static_cast<int>(pad_y)),
                                  resized.size())));
    scale = r;
    return canvas;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 5) {
        std::cerr << "usage: " << argv[0]
                  << " <model> <bank.json> <image> <predicates|csv> [boxes.txt]\n"
                  << "  predicates: 逗号分隔，如 beside,\"in front of\",behind,on\n"
                  << "  boxes.txt  : 可选，每行 \"name x1 y1 x2 y2 [conf] [cls_id]\"，"
                  << "坐标需在 448 letterbox 空间\n";
        return 2;
    }

    hal::RelationConfig cfg;
    cfg.model_path = argv[1];
    cfg.predicate_bank_path = argv[2];
    cfg.img_size = 448;
    cfg.max_boxes = 32;
    cfg.batch_size = 1;
    cfg.precision = "fp32";   // engine 是 trtexec 默认 FP16 建的，这里只验加载
    cfg.score_threshold = 0.05f;   // 放宽，便于和 Python 全量结果对照
    cfg.max_relations = 64;

    {
        std::stringstream ss(argv[4]);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty()) cfg.predicates.push_back(tok);
        }
    }

    // 后端内部用 LOG_ERROR_FMT 报错，测试进程里日志系统没初始化的话
    // 失败只会表现为一句空的 "infer 失败:"，看不到原因。
    LogManager::get_instance().initialize();

    std::cout << "=== 1. 选后端并加载 ===\n";
    auto engine = hal::RelationFactory::instance().create(hal::RelationBackend::AUTO);
    if (!engine) {
        std::cerr << "没有可用后端\n";
        return 1;
    }
    std::cout << "  后端: " << engine->getBackendName() << "\n";

    std::string err;
    if (!engine->loadModel(cfg)) {
        std::cerr << "loadModel 失败: " << err << "\n";
        return 1;
    }
    const auto sz = engine->getInputSize();
    std::printf("  输入 %dx%d\n", sz.first, sz.second);

    std::cout << "=== 2. 激活谓词 ===\n";
    if (!cfg.predicates.empty()) {
        if (!engine->setPredicates(cfg.predicates)) {
            std::cerr << "setPredicates 失败: " << err << "\n";
            return 1;
        }
    }
    const auto preds = engine->getPredicates();
    std::printf("  生效谓词 %zu 个:", preds.size());
    for (const auto& p : preds) std::printf(" %s", p.c_str());
    std::printf("\n");
    std::printf("  score_threshold = %.3f\n", engine->getScoreThreshold());

    std::cout << "=== 3. 读图 + letterbox ===\n";
    cv::Mat img = cv::imread(argv[3], cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "读图失败: " << argv[3] << "\n";
        return 1;
    }
    float scale = 1.0f, pad_x = 0.0f, pad_y = 0.0f;
    cv::Mat lb = letterbox(img, sz.first, scale, pad_x, pad_y);
    std::printf("  %dx%d -> %dx%d  scale=%.4f pad=(%.1f,%.1f)\n",
                img.cols, img.rows, lb.cols, lb.rows, scale, pad_x, pad_y);

    std::cout << "=== 4. 组装框 ===\n";
    hal::RelationInput in;
    in.image = lb.data;
    in.image_width = lb.cols;
    in.image_height = lb.rows;
    in.image_stride = static_cast<int>(lb.step);
    in.is_bgr = true;
    // test5.jpg（700x393）在 448 letterbox 空间的框，由原图坐标经
    // (xy - pad)/scale 换算而来，与 Python 端用同一组值以便对照。
    in.boxes = {{ 137.5f, -128.1f, 781.2f, 757.8f },
                { 418.8f,  -64.1f, 542.2f, 762.5f }};
    in.boxes[0].confidence = 1.00f;  in.boxes[0].class_id = 1;  // wall
    in.boxes[1].confidence = 0.27f;  in.boxes[1].class_id = 0;  // child
    in.class_names = { "child", "wall", "gate" };

    // boxes.txt 可覆盖默认框（坐标在 448 空间）
    if (argc > 5) {
        std::ifstream f(argv[5]);
        if (!f) {
            std::cerr << "读 boxes 失败: " << argv[5] << "\n";
            return 1;
        }
        in.boxes.clear();
        std::string line;
        while (std::getline(f, line)) {
            if (line.empty()) continue;
            std::istringstream ls(line);
            hal::RelationInput::Box b;
            int cls = -1;
            ls >> b.x1 >> b.y1 >> b.x2 >> b.y2 >> b.confidence >> cls;
            if (!ls) continue;
            b.class_id = cls;
            in.boxes.push_back(b);
        }
    }
    std::printf("  %zu 个框:", in.boxes.size());
    for (const auto& b : in.boxes)
        std::printf(" [%s (%.0f,%.0f)-(%.0f,%.0f) c=%.2f]",
                    in.class_names[b.class_id].c_str(),
                    b.x1, b.y1, b.x2, b.y2, b.confidence);
    std::printf("\n");

    std::cout << "=== 5. 推理 ===\n";
    std::vector<hal::RelationTriplet> out;
    const auto t0 = std::chrono::steady_clock::now();
    if (!engine->infer(in, out)) {
        std::cerr << "infer 失败: " << err << "\n";
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("  耗时 %.1f ms，输出 %zu 条\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count(),
                out.size());

    std::cout << "=== 6. 结果（供对比）===\n";
    // 空结果时把阈值压到 0.001 再跑一次，用来区分"分数太低"和"图没输出配对"。
    // 两种情况的排查方向完全不同：前者调阈值，后者查框坐标格式/box_counts。
    if (out.empty()) {
        engine->setScoreThreshold(0.001f);
        std::vector<hal::RelationTriplet> probe;
        engine->infer(in, probe);
        if (probe.empty()) {
            std::printf("  阈值 0.001 仍为 0 条 -> 不是阈值问题。查：\n"
                        "    - boxes 必须是**归一化 cxcywh(0..1)**，不是 xyxy 像素\n"
                        "    - box_counts 与实际框数一致\n"
                        "    - 后端日志里 valid= 是否 > 0\n");
        } else {
            std::printf("  阈值 0.001 下有 %zu 条，说明只是分数低于当前阈值:\n",
                        probe.size());
            for (size_t i = 0; i < probe.size() && i < 8; ++i) {
                const std::string s =
                    in.class_names[in.boxes[probe[i].subject_index].class_id];
                const std::string o =
                    in.class_names[in.boxes[probe[i].object_index].class_id];
                std::printf("    %-6s -> %-6s : %-16s %.6f\n", s.c_str(), o.c_str(),
                            probe[i].predicate.c_str(), probe[i].score);
            }
        }
    }
    for (const auto& t : out) {
        const std::string s =
            in.class_names[in.boxes[t.subject_index].class_id];
        const std::string o =
            in.class_names[in.boxes[t.object_index].class_id];
        std::printf("  %-6s -> %-6s : %-16s %.6f\n",
                    s.c_str(), o.c_str(), t.predicate.c_str(), t.score);
    }
    std::printf("TOTAL %zu\n", out.size());

    engine->unload();
    return 0;
}