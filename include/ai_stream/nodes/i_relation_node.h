// include/ai_stream/nodes/i_relation_node.h
#pragma once

#include "ai_stream/core/node.h"
#include "ai_stream/core/packet.h"
#include "ai_stream/hal/relation_factory.h"

#include <string>
#include <vector>

namespace ai_stream {
namespace nodes {

/**
 * @brief 视觉关系节点接口
 *
 * 消费上游的检测结果（InferenceResultPacket::detections），对「主体-谓词-客体」
 * 三元组打分，结果写入 packet->relations。
 *
 * 与动作识别节点不同，关系模型**不自己检测**，所以它不需要 clip/滑窗；
 * 但它需要知道边界类别（wall/gate 之类），因为只有主体是儿童、客体是边界时
 * 才构成一条可告警的关系。
 */
class IRelationNode : public core::Node {
public:
    using core::Node::Node;
    virtual ~IRelationNode() = default;

    /** 关系模型路径（.engine / .onnx） */
    virtual void setModelPath(const std::string& path) = 0;

    /** 谓词表 JSON（阈值 / 逐谓词阈值 / W / alpha） */
    virtual void setPredicateBank(const std::string& path) = 0;

    /** 关系图的输入边长；0 表示用模型自带的 img_size */
    virtual void setImageSize(int size) = 0;

    /** 运行期激活的谓词。空 = 用引擎自带的默认谓词集 */
    virtual void setPredicates(const std::vector<std::string>& predicates) = 0;

    /** 关系分门限 */
    virtual void setScoreThreshold(float threshold) = 0;

    /** 参与关系判定的边界类别，如 {"wall", "gate"} */
    virtual void setBoundaryClasses(const std::vector<std::string>& classes) = 0;

    /** 框置信度低于此值的不参与打分，避免把抖动框喂给关系模型 */
    virtual void setBoxConfidence(float min_conf) = 0;

    /** 每帧最多送入关系模型的框数；0 表示用模型自带 max_boxes */
    virtual void setMaxBoxes(int max_boxes) = 0;

    virtual hal::RelationBackend getBackend() const = 0;

    /** 最近一次推理的耗时，填进 packet->cost_time_map 便于现场排查 */
    virtual int64_t getLastInferMs() const = 0;
};

} // namespace nodes
} // namespace ai_stream
