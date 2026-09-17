// tests/unit/nodes/test_falling_rule.cpp
// FallingRule 跌倒过程规则测试：基于 track_id 的 person->down 类别转移 + down 持续时长
#include <gtest/gtest.h>
#include "rules/alert/falling_rule.h"
#include "ai_stream/core/packet.h"
#include <memory>
#include <string>
#include <vector>

using namespace ai_stream;
using namespace ai_stream::rules;
using namespace ai_stream::core;

namespace {

std::shared_ptr<InferenceResultPacket> makePacket(
    int64_t ts, std::vector<InferenceResultPacket::BBox> dets) {
    auto p = std::make_shared<InferenceResultPacket>();
    p->timestamp_ms = ts;
    p->frame_id = ts;
    p->detections = std::move(dets);
    return p;
}

InferenceResultPacket::BBox makeBox(int track_id, const std::string& cls) {
    InferenceResultPacket::BBox b;
    b.x = 0; b.y = 0; b.w = 50; b.h = 100;
    b.confidence = 0.9f;
    b.class_id = 0;
    b.class_name = cls;
    b.track_id = track_id;
    return b;
}

int countOccur(const AlertResult& r) {
    int n = 0;
    for (const auto& e : r.alert_events) {
        if (e.status == AlertStatus::ALERT_STATUS_OCCUR) n++;
    }
    return n;
}

bool hasObject(const AlertResult& r, int track_id) {
    for (const auto& e : r.alert_events) {
        for (int id : e.object_ids) {
            if (id == track_id) return true;
        }
    }
    return false;
}

} // namespace

// person 站立后转为 down，且 down 持续达标 -> 告警
TEST(FallingRuleTest, PersonThenDownPersistTriggersAlert) {
    FallingRule rule;
    nlohmann::json cfg = {
        {"name", "falling"},
        {"down_confirm_ms", 100},
        {"alert_duration_ms", 0},
        {"track_timeout_ms", 10000}
    };
    ASSERT_TRUE(rule.initialize(cfg));

    auto fire = [&](int64_t ts, const std::string& cls) {
        AlertResult r;
        rule.process(makePacket(ts, {makeBox(1, cls)}), r, ts);
        return r;
    };

    fire(0, "person");
    fire(50, "person");

    // down 开始，尚未持续达标
    EXPECT_TRUE(fire(100, "down").alert_events.empty());
    EXPECT_TRUE(fire(150, "down").alert_events.empty());

    // down 持续达标（>=100ms），事件建立
    fire(250, "down");

    // 确认后 duration>alert_duration_ms_ -> 状态提升为 OCCUR（当帧 push 循环已过，下一帧上报）
    fire(300, "down");
    AlertResult r = fire(350, "down");
    EXPECT_EQ(countOccur(r), 1);
    EXPECT_TRUE(hasObject(r, 1));
}

// 从未观测到 person 的 down 不告警（要求完整转移过程）
TEST(FallingRuleTest, DownWithoutPriorPersonNoAlert) {
    FallingRule rule;
    ASSERT_TRUE(rule.initialize({{"down_confirm_ms", 100}, {"alert_duration_ms", 0}}));

    for (int64_t t = 0; t <= 1000; t += 100) {
        AlertResult r;
        rule.process(makePacket(t, {makeBox(2, "down")}), r, t);
        EXPECT_TRUE(r.alert_events.empty()) << "t=" << t;
    }
}

// down 过程中恢复为 person -> 取消本次跌倒，不应告警
TEST(FallingRuleTest, RecoverToPersonCancelsFall) {
    FallingRule rule;
    ASSERT_TRUE(rule.initialize({{"down_confirm_ms", 100}, {"alert_duration_ms", 0}}));

    auto fire = [&](int64_t ts, const std::string& cls) {
        AlertResult r;
        rule.process(makePacket(ts, {makeBox(3, cls)}), r, ts);
        return r;
    };

    fire(0, "person");
    fire(50, "down");    // 开始 down
    fire(100, "person"); // 恢复站立，取消

    bool alerted = false;
    for (int64_t t = 150; t <= 1000; t += 50) {
        auto r = fire(t, "person");
        if (countOccur(r) > 0) alerted = true;
    }
    EXPECT_FALSE(alerted);
}

// 自定义 down 类别名（数组）
TEST(FallingRuleTest, CustomDownClassNames) {
    FallingRule rule;
    nlohmann::json cfg = {
        {"down_class", {"lying", "fall_down"}},
        {"down_confirm_ms", 50},
        {"alert_duration_ms", 0}
    };
    ASSERT_TRUE(rule.initialize(cfg));

    auto fire = [&](int64_t ts, const std::string& cls) {
        AlertResult r;
        rule.process(makePacket(ts, {makeBox(4, cls)}), r, ts);
        return r;
    };

    fire(0, "person");
    fire(100, "lying");   // 转移
    fire(200, "lying");   // 持续达标，事件建立
    fire(250, "lying");   // 提升为 OCCUR
    AlertResult r = fire(300, "lying");
    EXPECT_EQ(countOccur(r), 1);
    EXPECT_TRUE(hasObject(r, 4));
}

// getType 复用 FALL_DOWN
TEST(FallingRuleTest, TypeIsFallDown) {
    FallingRule rule;
    EXPECT_EQ(rule.getType(), AlertType::FALL_DOWN);
    EXPECT_EQ(rule.getTypeName(), "fall_down");
}
