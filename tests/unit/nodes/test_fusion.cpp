// tests/unit/nodes/test_fusion.cpp
// FusionNode 多推理源检测框融合 + 动作融合兼容性测试
#include <gtest/gtest.h>
#include "fusion/fusion_node.h"
#include "ai_stream/core/packet.h"
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace ai_stream;
using namespace ai_stream::core;
using namespace ai_stream::nodes;

namespace {

// 记录下游收到的包
class RecordingNode : public Node {
public:
    explicit RecordingNode(const std::string& name) : Node(name) {}
    bool start() override { running_ = true; return true; }
    void stop() override { running_ = false; }
    void pushData(std::shared_ptr<BasePacket> packet) override {
        std::lock_guard<std::mutex> lock(m_);
        received_.push_back(packet);
    }
    std::vector<std::shared_ptr<BasePacket>> received() {
        std::lock_guard<std::mutex> lock(m_);
        return received_;
    }
    size_t count() {
        std::lock_guard<std::mutex> lock(m_);
        return received_.size();
    }
    void clear() {
        std::lock_guard<std::mutex> lock(m_);
        received_.clear();
    }
private:
    std::mutex m_;
    std::vector<std::shared_ptr<BasePacket>> received_;
};

std::shared_ptr<InferenceResultPacket> makeDetResult(
    const std::string& producer, uint32_t stream_id, int64_t frame_id,
    const std::vector<std::pair<int, std::string>>& classes) {
    auto r = std::make_shared<InferenceResultPacket>();
    r->producer_id = producer;
    r->stream_id = stream_id;
    r->frame_id = frame_id;
    r->timestamp_ms = 1000 + frame_id;
    for (const auto& [id, name] : classes) {
        InferenceResultPacket::BBox box;
        box.x = 10; box.y = 10; box.w = 50; box.h = 50;
        box.confidence = 0.9f;
        box.class_id = id;
        box.class_name = name;
        r->detections.push_back(box);
    }
    return r;
}

template<typename Pred>
bool waitUntil(Pred pred, int timeout_ms = 3000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    return pred();
}

struct FusionFixture : public ::testing::Test {
    std::shared_ptr<FusionNodeImpl> fusion;
    std::shared_ptr<RecordingNode> sink;

    void setup(const nlohmann::json& params) {
        fusion = std::make_shared<FusionNodeImpl>();
        ASSERT_TRUE(fusion->configure("fusion1", params));
        sink = std::make_shared<RecordingNode>("sink");
        fusion->addDownstream(sink);
        ASSERT_TRUE(sink->start());
        ASSERT_TRUE(fusion->start());
    }

    void TearDown() override {
        if (fusion) fusion->stop();
        if (sink) sink->stop();
    }
};

} // namespace

// 双源同帧到齐 → 合并广播一次，无偏移时 id/name 原样
TEST_F(FusionFixture, MergeTwoSourcesNoOffset) {
    setup({
        {"mode", "detection_merge"},
        {"detection_sources", {"infer1", "infer2"}},
        {"wait_timeout_ms", 500}
    });

    fusion->pushData(makeDetResult("infer1", 1, 100, {{0, "person"}, {1, "helmet"}}));
    fusion->pushData(makeDetResult("infer2", 1, 100, {{0, "crystal"}, {1, "fire"}}));

    ASSERT_TRUE(waitUntil([&] { return sink->count() >= 1; }));
    auto pkts = sink->received();
    ASSERT_EQ(pkts.size(), 1u);
    auto merged = std::dynamic_pointer_cast<InferenceResultPacket>(pkts[0]);
    ASSERT_NE(merged, nullptr);
    ASSERT_EQ(merged->detections.size(), 4u);

    // 无偏移：class_id 原样（两源各自 0/1），class_name 保留
    std::vector<std::string> names;
    for (const auto& d : merged->detections) names.push_back(d.class_name);
    EXPECT_NE(std::find(names.begin(), names.end(), "person"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "helmet"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "crystal"), names.end());
    EXPECT_NE(std::find(names.begin(), names.end(), "fire"), names.end());
}

// 配置 class_offsets：对应来源 class_id 加偏移，class_name 不变
TEST_F(FusionFixture, MergeWithClassOffset) {
    setup({
        {"mode", "detection_merge"},
        {"detection_sources", {"infer1", "infer2"}},
        {"class_offsets", {{"infer1", 0}, {"infer2", 14}}},
        {"wait_timeout_ms", 500}
    });

    fusion->pushData(makeDetResult("infer1", 1, 7, {{0, "person"}}));
    fusion->pushData(makeDetResult("infer2", 1, 7, {{0, "crystal"}, {2, "smoke"}}));

    ASSERT_TRUE(waitUntil([&] { return sink->count() >= 1; }));
    auto merged = std::dynamic_pointer_cast<InferenceResultPacket>(sink->received()[0]);
    ASSERT_NE(merged, nullptr);
    ASSERT_EQ(merged->detections.size(), 3u);

    for (const auto& d : merged->detections) {
        if (d.class_name == "person") {
            EXPECT_EQ(d.class_id, 0);        // infer1 偏移 0
        } else if (d.class_name == "crystal") {
            EXPECT_EQ(d.class_id, 14);       // infer2 偏移 14
        } else if (d.class_name == "smoke") {
            EXPECT_EQ(d.class_id, 16);       // 2 + 14
        }
    }
}

// 单源超时 → 部分合并广播
TEST_F(FusionFixture, PartialMergeOnTimeout) {
    setup({
        {"mode", "detection_merge"},
        {"detection_sources", {"infer1", "infer2"}},
        {"wait_timeout_ms", 50}
    });

    // 只有 infer1 到达，infer2 缺失
    fusion->pushData(makeDetResult("infer1", 1, 42, {{0, "person"}}));

    ASSERT_TRUE(waitUntil([&] { return sink->count() >= 1; }, 2000));
    auto merged = std::dynamic_pointer_cast<InferenceResultPacket>(sink->received()[0]);
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->detections.size(), 1u);   // 部分合并：仅 infer1 的框
    EXPECT_EQ(merged->frame_id, 42);
}

// 未配置来源的结果包原样透传
TEST_F(FusionFixture, UnknownSourcePassthrough) {
    setup({
        {"mode", "detection_merge"},
        {"detection_sources", {"infer1", "infer2"}},
        {"wait_timeout_ms", 500}
    });

    auto pkt = makeDetResult("other_infer", 1, 5, {{0, "person"}});
    fusion->pushData(pkt);

    ASSERT_TRUE(waitUntil([&] { return sink->count() >= 1; }));
    auto got = std::dynamic_pointer_cast<InferenceResultPacket>(sink->received()[0]);
    ASSERT_NE(got, nullptr);
    EXPECT_EQ(got.get(), pkt.get());   // 原包透传
}

// 动作融合兼容：action 缓存附加到 detection 包（原 FRAME_LEVEL 行为）
TEST_F(FusionFixture, LegacyActionFusion) {
    setup({{"mode", "action"}, {"timestamp_threshold_ms", 1000}});

    // 先推动作识别结果（建立缓存）
    auto action = std::make_shared<InferenceResultPacket>();
    action->producer_id = "action1";
    action->timestamp_ms = 2000;
    InferenceResultPacket::ActionResult ar;
    ar.action_label = "climbing";
    ar.confidence = 0.8f;
    action->action_results.push_back(ar);
    fusion->pushData(action);

    // 再推检测结果（时间戳接近，应附加动作）
    auto det = makeDetResult("tracker1", 1, 1, {{0, "person"}});
    det->timestamp_ms = 2100;
    fusion->pushData(det);

    ASSERT_TRUE(waitUntil([&] { return sink->count() >= 1; }));
    auto got = std::dynamic_pointer_cast<InferenceResultPacket>(sink->received()[0]);
    ASSERT_NE(got, nullptr);
    ASSERT_EQ(got->detections.size(), 1u);
    ASSERT_EQ(got->action_results.size(), 1u);
    EXPECT_EQ(got->action_results[0].action_label, "climbing");
}

// detection_merge 模式下也兼容动作结果附加
TEST_F(FusionFixture, MergeModeAlsoFusesAction) {
    setup({
        {"mode", "detection_merge"},
        {"detection_sources", {"infer1"}},
        {"wait_timeout_ms", 500},
        {"timestamp_threshold_ms", 1000}
    });

    auto action = std::make_shared<InferenceResultPacket>();
    action->producer_id = "action1";
    action->timestamp_ms = 5000;
    InferenceResultPacket::ActionResult ar;
    ar.action_label = "fighting";
    ar.confidence = 0.7f;
    action->action_results.push_back(ar);
    fusion->pushData(action);

    auto det = makeDetResult("infer1", 1, 9, {{0, "person"}});
    det->timestamp_ms = 5100;
    fusion->pushData(det);

    ASSERT_TRUE(waitUntil([&] { return sink->count() >= 1; }));
    auto merged = std::dynamic_pointer_cast<InferenceResultPacket>(sink->received()[0]);
    ASSERT_NE(merged, nullptr);
    ASSERT_EQ(merged->detections.size(), 1u);
    ASSERT_EQ(merged->action_results.size(), 1u);
    EXPECT_EQ(merged->action_results[0].action_label, "fighting");
}

// 缺少 detection_sources 配置 → configure 失败
TEST(FusionConfigTest, MissingSourcesFails) {
    auto fusion = std::make_shared<FusionNodeImpl>();
    EXPECT_FALSE(fusion->configure("fusion1", {{"mode", "detection_merge"}}));
}

// 跨源 NMS：同名类别高 IoU 框去重
TEST_F(FusionFixture, CrossNmsDedup) {
    setup({
        {"mode", "detection_merge"},
        {"detection_sources", {"infer1", "infer2"}},
        {"wait_timeout_ms", 500},
        {"cross_nms", true},
        {"nms_iou_threshold", 0.5}
    });

    // 两个来源在同一位置都检测出 "person"（IoU=1），应去重为 1
    auto r1 = makeDetResult("infer1", 1, 3, {{0, "person"}});
    auto r2 = makeDetResult("infer2", 1, 3, {{0, "person"}});
    r1->detections[0].confidence = 0.9f;
    r2->detections[0].confidence = 0.8f;
    fusion->pushData(r1);
    fusion->pushData(r2);

    ASSERT_TRUE(waitUntil([&] { return sink->count() >= 1; }));
    auto merged = std::dynamic_pointer_cast<InferenceResultPacket>(sink->received()[0]);
    ASSERT_NE(merged, nullptr);
    EXPECT_EQ(merged->detections.size(), 1u);
    EXPECT_FLOAT_EQ(merged->detections[0].confidence, 0.9f);  // 保留高置信度
}
