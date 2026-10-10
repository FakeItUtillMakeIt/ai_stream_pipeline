// src/nodes/infer/relation_recognition.h
#pragma once

#include "ai_stream/core/queued_node.h"
#include "ai_stream/hal/i_relation.h"
#include "ai_stream/hal/relation_factory.h"
#include "ai_stream/nodes/i_relation_node.h"

#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace ai_stream {
namespace nodes {

/**
 * @brief 视觉关系节点（RelateAnything 场景图模型）
 *
 * 上游是 tracker 之后的 InferenceResultPacket。节点做三件事：
 *   1. 从 source_frame 取原图，按关系图要求的边长做 letterbox；
 *   2. 把 detections 映射到 letterbox 坐标（框要留在同一空间里）；
 *   3. 一次前向拿到全部三元组，写进 packet->relations。
 *
 * 为什么自己 letterbox 而不是复用上游的：上游 preprocess 面向检测模型（640），
 * 关系图是 448，两者的缩放比例和 padding 不同。用上游那份变换会把框映射错位，
 * 而错位不报错，只会表现为"关系分莫名偏低"。
 */
class RelationRecognitionNode : public core::QueuedNode<IRelationNode> {
public:
    struct Config {
        std::string model_path;
        std::string predicate_bank_path;
        int image_size = 0;                       // 0 = 用模型自带
        int max_boxes = 0;                        // 0 = 用模型自带
        int batch_size = 1;
        std::string precision = "fp32";
        std::vector<std::string> predicates;
        //! 逐谓词门限覆盖（谓词, 阈值）；不要照抄模型 bank 的标定值
        std::vector<std::pair<std::string, float>> per_predicate_threshold;
        std::vector<std::string> boundary_classes{"wall", "gate"};
        float score_threshold = 0.2f;
        float box_confidence = 0.25f;
        int max_relations = 256;
        int device_id = 0;
        // 隔帧跑关系推理（>1 时降 GPU 负载）。跳过的帧不跑推理、relations 留空，
        // 下游规则靠 max_disappear_count 容忍这些空帧。child 贴墙不是亚秒级事件，
        // 设 2~3 通常无感，却能显著抬高整链 fps。
        int run_every_n_frames = 1;
    };

    RelationRecognitionNode();
    ~RelationRecognitionNode() override;

    bool onStartup() override;
    void onShutdown() override;
    void processPacket(std::shared_ptr<core::BasePacket> packet) override;
    bool configureImpl(const std::string& node_id, const nlohmann::json& params) override;

    // IRelationNode
    void setModelPath(const std::string& path) override;
    void setPredicateBank(const std::string& path) override;
    void setImageSize(int size) override;
    void setPredicates(const std::vector<std::string>& predicates) override;
    void setScoreThreshold(float threshold) override;
    void setBoundaryClasses(const std::vector<std::string>& classes) override;
    void setBoxConfidence(float min_conf) override;
    void setMaxBoxes(int max_boxes) override;
    hal::RelationBackend getBackend() const override { return backend_; }
    int64_t getLastInferMs() const override { return last_infer_ms_; }

private:
    bool loadEngine();
    /** letterbox 到 image_size，返回 scale/pad 供框坐标换算 */
    cv::Mat letterbox(const cv::Mat& src, float& scale, int& pad_x, int& pad_y) const;
    void fillTriplets(const core::InferenceResultPacket& in,
                      const std::vector<hal::RelationTriplet>& triplets,
                      const std::vector<std::string>& all_classes,
                      const std::vector<int>& box_track_ids,
                      std::vector<core::InferenceResultPacket::RelationResult>& out);

    Config cfg_;
    hal::RelationBackend backend_ = hal::RelationBackend::AUTO;
    hal::RelationEnginePtr engine_;
    std::string node_id_;
    bool is_initialized_ = false;
    int image_size_ = 448;
    int64_t last_infer_ms_ = 0;
    int64_t frame_counter_ = 0;   // 用于 run_every_n_frames 隔帧
    std::mutex engine_mutex_;
};

} // namespace nodes
} // namespace ai_stream
