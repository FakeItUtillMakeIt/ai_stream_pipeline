// include/ai_stream/hal/i_relation.h
// 视觉关系抽象接口——隔离 TensorRT / ONNXRuntime 等后端
//
// 与检测引擎的关键差别：关系模型**不检测物体**，它消费上游已经检出的区域，
// 对这些区域两两打分。所以 infer() 的输入除图像外还有框，且阈值与谓词表是
// 运行期可换的（谓词向量作为图输入，不需要重新导出引擎）。
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace ai_stream {
namespace hal {

/**
 * @brief 一条候选关系（引擎输出，未过滤）
 *
 * box_index 指向调用方传入的框数组下标。subject_index / object_index 由引擎
 * 从 valid_mask 与 sub_idx/obj_idx 解出，-1 表示该侧无有效配对。
 */
struct RelationTriplet {
    int subject_index = -1;
    int object_index = -1;
    std::string predicate;
    float score = 0.0f;          // 标定后关系分，量纲与阈值一致
    int predicate_index = -1;    // 在谓词表中的下标
};

/**
 * @brief 关系引擎配置
 */
struct RelationConfig {
    std::string model_path;          // .engine 或 .onnx
    std::string predicate_bank_path; // 谓词表 JSON（阈值 / 逐谓词阈值 / W / alpha）
    std::string config_path;         // 可选：与 model_path 同目录的 relateanything.json

    int img_size = 448;              // 关系图输入边长（DINOv3 主干已烤进图内）
    int max_boxes = 32;              // 每帧送入的最大框数，超出按置信度截断
    int batch_size = 1;
    std::string precision = "fp32";  // 导出的引擎是 FP32，别默认 fp16

    // 运行期激活的谓词。留空则用引擎内置的默认谓词集。
    std::vector<std::string> predicates;

    // 单条关系分门限，在引擎内过滤，避免把上千条候选搬到 host
    float score_threshold = 0.2f;

    /**
     * 逐谓词门限覆盖。
     *
     * 只在"某个谓词太吵、想单独压它"时才用。**不要**拿模型 bank 里的标定值
     * 来填：那些是作者标的工作点（如 beside 0.980），而实测关系分常在 0.4 量级，
     * 照搬会让规则永远静默且不报错。
     */
    std::vector<std::pair<std::string, float>> per_predicate_threshold;
    int max_relations = 256;         // 每帧返回上限，防止 host 侧被打爆

    int device_id = 0;
};

/**
 * @brief 关系引擎输入
 *
 * boxes 用 letterbox 之后的像素坐标（cxcywh 或 xyxy 由 boxes_format 决定）。
 * 变换参数由调用方算出——本接口不假设图像做过什么缩放。
 */
struct RelationInput {
    const uint8_t* image = nullptr;   // 已 letterbox 到 img_size 的数据
    int image_width = 0;
    int image_height = 0;
    int image_stride = 0;             // 字节/行；0 表示按 width * 3 紧密排列
    bool is_bgr = true;               // true=BGR，false=RGB

    struct Box {
        float x1 = 0, y1 = 0, x2 = 0, y2 = 0;
        float confidence = 1.0f;    // 检测置信度；参与关系排序折扣，也供规则做门控
        int class_id = -1;
        int track_id = -1;
    };
    std::vector<Box> boxes;           // xyxy，像素坐标，已在 letterbox 空间
    std::vector<std::string> class_names; // 与 class_id 对齐；可空，只按 id 用
};

/**
 * @brief 视觉关系引擎抽象接口
 *
 * 封装场景图关系模型（RelateAnything 等）：吃图像 + 区域，吐三元组。
 */
class IRelationEngine {
public:
    virtual ~IRelationEngine() = default;

    /**
     * @brief 加载引擎与谓词表
     */
    virtual bool loadModel(const RelationConfig& config) = 0;

    /**
     * @brief 对一批区域打分
     * @param out 已按 score_threshold 过滤并按分数降序
     * @return 成功返回 true；框数为 0 时也返回 true 且 out 为空
     */
    virtual bool infer(const RelationInput& input,
                       std::vector<RelationTriplet>& out) = 0;

    /**
     * @brief 推理已预处理的 NCHW float 数据（上游已完成 letterbox+normalize）
     * @return 后端不支持时返回 false
     */
    virtual bool inferPreprocessed(const float* input_nchw, size_t size_bytes,
                                   const RelationInput& meta,
                                   std::vector<RelationTriplet>& out)
    {
        (void)input_nchw; (void)size_bytes; (void)meta; (void)out;
        return false;
    }

    virtual void unload() = 0;

    virtual std::pair<int, int> getInputSize() const = 0;

    /** 运行期当前生效的谓词表，规则侧要显示"用哪个谓词判的"就取这里 */
    virtual std::vector<std::string> getPredicates() const = 0;

    /** 换谓词子集。不需要重新加载引擎——谓词向量是图输入。 */
    virtual bool setPredicates(const std::vector<std::string>& predicates) = 0;

    virtual float getScoreThreshold() const = 0;
    virtual bool setScoreThreshold(float threshold) = 0;

    /** 工厂 AUTO 选择要靠它和 isAvailable() 挑后端 */
    virtual std::string getBackendName() const = 0;
    virtual bool isAvailable() const = 0;
};

using RelationEnginePtr = std::unique_ptr<IRelationEngine>;

} // namespace hal
} // namespace ai_stream
