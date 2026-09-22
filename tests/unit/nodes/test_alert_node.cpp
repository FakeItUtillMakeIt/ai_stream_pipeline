// tests/unit/nodes/test_alert_node.cpp
// AlertNode 常驻线程池生命周期与并行规则回归测试
#include <gtest/gtest.h>
#include "alert/alert_node.h"
#include "ai_stream/core/packet.h"
#include <chrono>
#include <memory>
#include <thread>

using namespace ai_stream;
using namespace ai_stream::core;
using namespace ai_stream::nodes;

namespace {

std::shared_ptr<InferenceResultPacket> makePacket(int64_t frame_id) {
    auto p = std::make_shared<InferenceResultPacket>();
    p->stream_id = 1;
    p->frame_id = frame_id;
    p->timestamp_ms = frame_id * 40;
    return p;
}

} // namespace

// 并行模式：线程池创建/处理/停止不崩溃，且重复 start/stop 安全
TEST(AlertNodePoolTest, ParallelStartProcessStop) {
    auto node = std::make_shared<AlertNode>();
    nlohmann::json params = {
        {"process_type", "parallel"},
        {"rules", nlohmann::json::array({
            {{"type", "person_intrusion"}, {"params", {{"name", "pi"}}}},
            {{"type", "human_gathering"},  {"params", {{"name", "hg"}}}}
        })}
    };
    ASSERT_TRUE(node->configure("alert1", params));
    ASSERT_TRUE(node->start());

    for (int i = 0; i < 20; ++i) {
        node->pushData(makePacket(i));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    node->stop();
    EXPECT_FALSE(node->isRunning());

    // 自停后可再次启动（线程池需正确重建）
    ASSERT_TRUE(node->start());
    node->pushData(makePacket(100));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    node->stop();
    SUCCEED();
}

// 串行模式同样可用
TEST(AlertNodePoolTest, SequenceModeWorks) {
    auto node = std::make_shared<AlertNode>();
    nlohmann::json params = {
        {"process_type", "sequence"},
        {"rules", nlohmann::json::array({
            {{"type", "person_intrusion"}, {"params", {{"name", "pi"}}}}
        })}
    };
    ASSERT_TRUE(node->configure("alert2", params));
    ASSERT_TRUE(node->start());
    for (int i = 0; i < 5; ++i) node->pushData(makePacket(i));
    std::this_thread::sleep_for(std::chrono::milliseconds(30));
    node->stop();
    SUCCEED();
}
