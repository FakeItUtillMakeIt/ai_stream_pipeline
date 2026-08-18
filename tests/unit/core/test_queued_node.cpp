// tests/unit/core/test_queued_node.cpp
#include <gtest/gtest.h>
#include "ai_stream/core/queued_node.h"
#include "ai_stream/core/metrics.h"
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

using namespace ai_stream;
using namespace ai_stream::core;

namespace {

std::shared_ptr<BasePacket> makePacket(int64_t frame_id) {
    auto p = std::make_shared<BasePacket>();
    p->type = PacketType::META_DATA;
    p->frame_id = frame_id;
    return p;
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

class RecordingNode : public QueuedNode<Node> {
public:
    explicit RecordingNode(const std::string& name, int process_delay_ms = 0)
        : QueuedNode<Node>(name), delay_ms_(process_delay_ms) {}

    std::vector<int64_t> processedIds() {
        std::lock_guard<std::mutex> lock(m_);
        return processed_;
    }
    size_t processedCount() {
        std::lock_guard<std::mutex> lock(m_);
        return processed_.size();
    }
    std::atomic<int> stream_end_count{0};

protected:
    void processPacket(std::shared_ptr<BasePacket> packet) override {
        if (packet->type == PacketType::STREAM_END) {
            stream_end_count++;
            stop();
            broadcast(packet);
            return;
        }
        if (delay_ms_ > 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms_));
        }
        std::lock_guard<std::mutex> lock(m_);
        processed_.push_back(packet->frame_id);
    }

private:
    std::mutex m_;
    std::vector<int64_t> processed_;
    int delay_ms_;
};

// 同步接收节点：记录收到的包（用于验证 broadcast）
class SyncSinkNode : public Node {
public:
    explicit SyncSinkNode(const std::string& name) : Node(name) {}
    bool start() override { running_ = true; return true; }
    void stop() override { running_ = false; }
    void pushData(std::shared_ptr<BasePacket> packet) override {
        std::lock_guard<std::mutex> lock(m_);
        received_.push_back(packet);
    }
    size_t receivedCount() {
        std::lock_guard<std::mutex> lock(m_);
        return received_.size();
    }
    std::vector<std::shared_ptr<BasePacket>> received() {
        std::lock_guard<std::mutex> lock(m_);
        return received_;
    }
private:
    std::mutex m_;
    std::vector<std::shared_ptr<BasePacket>> received_;
};

} // namespace

TEST(QueuedNodeTest, ProcessesPacketsInOrder) {
    auto node = std::make_shared<RecordingNode>("order_node");
    // 容量需大于总包数，避免触发丢帧策略
    nlohmann::json params = {{"queue", {{"capacity", 128}}}};
    ASSERT_TRUE(node->configure("order_node", params));
    ASSERT_TRUE(node->start());

    constexpr int kCount = 100;
    for (int i = 0; i < kCount; ++i) {
        node->pushData(makePacket(i));
    }

    ASSERT_TRUE(waitUntil([&] { return node->processedCount() == kCount; }));
    auto ids = node->processedIds();
    for (int i = 0; i < kCount; ++i) {
        EXPECT_EQ(ids[i], i);
    }
    node->stop();
}

TEST(QueuedNodeTest, PushDataBeforeStartIsDropped) {
    auto node = std::make_shared<RecordingNode>("not_running_node");
    node->pushData(makePacket(1));
    ASSERT_TRUE(node->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(node->processedCount(), 0u);
    node->stop();
}

TEST(QueuedNodeTest, QueueConfigParsedFromJson) {
    auto node = std::make_shared<RecordingNode>("cfg_node");
    nlohmann::json params = {
        {"queue", {{"capacity", 3}, {"drop_policy", "drop_oldest"}, {"push_timeout_ms", 42}}}
    };
    EXPECT_TRUE(node->configure("cfg_node", params));
    ASSERT_TRUE(node->start());

    // 处理延迟 20ms，容量 3：快速推 20 个包，drop_oldest 保证最新帧最终被处理
    auto slow = std::make_shared<RecordingNode>("slow_node", 20);
    nlohmann::json slow_params = {
        {"queue", {{"capacity", 3}, {"drop_policy", "drop_oldest"}}}
    };
    ASSERT_TRUE(slow->configure("slow_node", slow_params));
    ASSERT_TRUE(slow->start());
    for (int i = 0; i < 20; ++i) {
        slow->pushData(makePacket(i));
    }
    // 等待处理完成（队列残留 + 正在处理的）
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    auto ids = slow->processedIds();
    ASSERT_FALSE(ids.empty());
    // drop_oldest：最后处理的必须是最新帧 19
    EXPECT_EQ(ids.back(), 19);
    slow->stop();
    node->stop();
}

TEST(QueuedNodeTest, DropNewestRecordsDroppedPackets) {
    MetricsCollector::instance().resetAll();
    auto node = std::make_shared<RecordingNode>("drop_node", 20);
    nlohmann::json params = {
        {"queue", {{"capacity", 2}, {"drop_policy", "drop_newest"}}}
    };
    ASSERT_TRUE(node->configure("drop_node", params));
    ASSERT_TRUE(node->start());

    for (int i = 0; i < 30; ++i) {
        node->pushData(makePacket(i));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(200));

    auto metrics = MetricsCollector::instance().getNodeMetrics("", "drop_node");
    EXPECT_GT(metrics.dropped_packets, 0u);
    EXPECT_LT(node->processedCount(), 30u);
    node->stop();
}

TEST(QueuedNodeTest, RestartAfterStopWorks) {
    auto node = std::make_shared<RecordingNode>("restart_node");
    ASSERT_TRUE(node->start());
    node->pushData(makePacket(1));
    ASSERT_TRUE(waitUntil([&] { return node->processedCount() == 1; }));
    node->stop();
    EXPECT_FALSE(node->isRunning());

    ASSERT_TRUE(node->start());
    node->pushData(makePacket(2));
    ASSERT_TRUE(waitUntil([&] { return node->processedCount() == 2; }));
    auto ids = node->processedIds();
    EXPECT_EQ(ids[0], 1);
    EXPECT_EQ(ids[1], 2);
    node->stop();
}

TEST(QueuedNodeTest, StreamEndStopsNodeAndBroadcasts) {
    auto node = std::make_shared<RecordingNode>("stream_end_node");
    auto sink = std::make_shared<SyncSinkNode>("stream_end_sink");
    node->addDownstream(sink);
    ASSERT_TRUE(node->start());
    ASSERT_TRUE(sink->start());

    node->pushData(makePacket(1));
    auto end = std::make_shared<BasePacket>();
    end->type = PacketType::STREAM_END;
    node->pushData(end);

    ASSERT_TRUE(waitUntil([&] { return node->stream_end_count.load() == 1; }));
    ASSERT_TRUE(waitUntil([&] { return !node->isRunning(); }));
    EXPECT_EQ(sink->receivedCount(), 1u);
    EXPECT_EQ(sink->received()[0]->type, PacketType::STREAM_END);
}

TEST(QueuedNodeTest, StopJoinsWorkerAndDrainsNoMore) {
    auto node = std::make_shared<RecordingNode>("join_node", 5);
    ASSERT_TRUE(node->start());
    for (int i = 0; i < 10; ++i) {
        node->pushData(makePacket(i));
    }
    node->stop();  // 应阻塞直到 worker 退出（不 join 自身死锁）
    EXPECT_FALSE(node->isRunning());
    size_t count_after_stop = node->processedCount();
    node->pushData(makePacket(999));
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(node->processedCount(), count_after_stop);
}
