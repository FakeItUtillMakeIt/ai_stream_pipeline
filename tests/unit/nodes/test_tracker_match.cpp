// tests/unit/nodes/test_tracker_match.cpp
// TrackerNode class_name 匹配测试：多推理源 class_id 冲突场景下轨迹不错配
#include <gtest/gtest.h>
#include "track/tracker_node.h"
#include "ai_stream/core/packet.h"
#include <chrono>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

using namespace ai_stream;
using namespace ai_stream::core;
using namespace ai_stream::nodes;

namespace {

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
private:
    std::mutex m_;
    std::vector<std::shared_ptr<BasePacket>> received_;
};

InferenceResultPacket::BBox makeBox(float x, float y, int class_id, const std::string& name) {
    InferenceResultPacket::BBox box;
    box.x = x; box.y = y; box.w = 100; box.h = 100;
    box.confidence = 0.9f;
    box.class_id = class_id;
    box.class_name = name;
    box.track_id = -1;
    return box;
}

std::shared_ptr<InferenceResultPacket> makeFrame(int64_t frame_id,
    std::vector<InferenceResultPacket::BBox> dets) {
    auto r = std::make_shared<InferenceResultPacket>();
    r->stream_id = 1;
    r->frame_id = frame_id;
    r->timestamp_ms = frame_id * 40;
    r->detections = std::move(dets);
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

struct TrackerFixture : public ::testing::Test {
    std::shared_ptr<TrackerNode> tracker;
    std::shared_ptr<RecordingNode> sink;

    void SetUp() override {
        tracker = std::make_shared<TrackerNode>();
        nlohmann::json params = {
            {"tracker_type", "ocsort"},
            {"ocsort_config", {{"min_hits", 1}, {"det_thresh", 0.1}, {"max_age", 30}}}
        };
        ASSERT_TRUE(tracker->configure("tracker1", params));
        sink = std::make_shared<RecordingNode>("sink");
        tracker->addDownstream(sink);
        ASSERT_TRUE(sink->start());
        ASSERT_TRUE(tracker->start());
    }

    void TearDown() override {
        if (tracker) tracker->stop();
        if (sink) sink->stop();
    }

    std::shared_ptr<InferenceResultPacket> pushFrame(int64_t frame_id,
        std::vector<InferenceResultPacket::BBox> dets) {
        auto frame = makeFrame(frame_id, std::move(dets));
        tracker->pushData(frame);
        size_t expect = sink->count() + 1;
        waitUntil([&] { return sink->count() >= expect; });
        return frame;
    }
};

} // namespace

// 同类连续帧保持同一 track_id（名称匹配不影响正常跟踪）
TEST_F(TrackerFixture, SameClassKeepsTrackId) {
    auto f1 = pushFrame(1, {makeBox(0, 0, 0, "person")});
    ASSERT_GE(f1->detections[0].track_id, 0);    // min_hits=1：首帧即出轨迹

    auto f2 = pushFrame(2, {makeBox(2, 0, 0, "person")});
    ASSERT_GE(f2->detections[0].track_id, 0);

    auto f3 = pushFrame(3, {makeBox(4, 0, 0, "person")});
    EXPECT_EQ(f3->detections[0].track_id, f1->detections[0].track_id);
    EXPECT_EQ(f3->detections[0].track_id, f2->detections[0].track_id);
}

// class_id 冲突场景：person 与 crystal 共用 id 0。
// person 消失后在 crystal 轨迹位置重现时，不得继承 crystal 的 track_id
TEST_F(TrackerFixture, ClassNamePreventsCrossClassTrackInheritance) {
    // 帧1：person(0,0) 与 crystal(200,0)，class_id 同为 0
    pushFrame(1, {makeBox(0, 0, 0, "person"), makeBox(200, 0, 0, "crystal")});

    // 帧2：仅 crystal
    auto f2 = pushFrame(2, {makeBox(200, 0, 0, "crystal")});
    ASSERT_GE(f2->detections[0].track_id, 0);
    int crystal_track_id = f2->detections[0].track_id;

    // 帧3：crystal 消失，person 出现在 crystal 原位置（与 crystal 轨迹 IoU≈1）
    auto f3 = pushFrame(3, {makeBox(200, 0, 0, "person")});

    // 按 class_id 匹配会错误继承 crystal 的 track_id；
    // 按 class_name 匹配则拒绝（轨迹绑定 "crystal" ≠ "person"），track_id 为 -1
    EXPECT_NE(f3->detections[0].track_id, crystal_track_id);
    EXPECT_EQ(f3->detections[0].track_id, -1);
}

// 类别名为空时回退 class_id 匹配
TEST_F(TrackerFixture, EmptyNameFallsBackToClassId) {
    auto f1 = pushFrame(1, {makeBox(0, 0, 3, "")});
    auto f2 = pushFrame(2, {makeBox(2, 0, 3, "")});
    ASSERT_GE(f2->detections[0].track_id, 0);

    auto f3 = pushFrame(3, {makeBox(4, 0, 3, "")});
    EXPECT_EQ(f3->detections[0].track_id, f2->detections[0].track_id);
}
