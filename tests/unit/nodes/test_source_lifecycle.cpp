// tests/unit/nodes/test_source_lifecycle.cpp
// 源节点生命周期回归测试：
// worker 线程自停（EOF/重连失败）后，stop()/析构/重启 必须安全
// （历史 bug：自停后线程未 join，析构 joinable 线程触发 std::terminate）
#include <gtest/gtest.h>
#include "source/file_source.h"
#include "ai_stream/core/packet.h"
#include <chrono>
#include <memory>
#include <string>
#include <thread>

using namespace ai_stream;
using namespace ai_stream::nodes;

namespace {

std::string sampleVideoUrl() {
    return std::string("file://") + TEST_DATA_DIR + "/sample_5s.mp4";
}

template<typename Pred>
bool waitUntil(Pred pred, int timeout_ms = 15000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (std::chrono::steady_clock::now() < deadline) {
        if (pred()) return true;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return pred();
}

} // namespace

// EOF 自停（loop=false）后：stop/析构不崩溃，且可重启
TEST(FileSourceLifecycleTest, SelfStopThenStopDestroyRestart) {
    auto node = std::make_shared<FileSourceNode>();
    nlohmann::json params = {
        {"url", sampleVideoUrl()},
        {"loop", false},
        {"realtime", false}   // 全速读完，快速触发 EOF 自停
    };
    ASSERT_TRUE(node->configure("src_test", params));
    ASSERT_TRUE(node->start());
    EXPECT_TRUE(node->isRunning());

    // 等待 worker 读到 EOF 自行退出（running_ 被 worker 置 false）
    ASSERT_TRUE(waitUntil([&] { return !node->isRunning(); }));

    // 自停后 stop 必须安全（旧实现：提前 return，线程未 join）
    node->stop();

    // 自停后重启必须安全（旧实现：对 joinable 线程赋值 → std::terminate）
    ASSERT_TRUE(node->start());
    EXPECT_TRUE(node->isRunning());
    ASSERT_TRUE(waitUntil([&] { return !node->isRunning(); }));

    // 析构路径：reset 触发 ~FileSourceNode → stop → join，不得 terminate
    node.reset();
}

// EOF 自停后【不经过 stop】直接重启（用户真实场景）：
// 旧实现残留已打开的 fmt_ctx_，avformat_open_input 复用导致段错误
TEST(FileSourceLifecycleTest, RestartDirectlyAfterSelfStop) {
    auto node = std::make_shared<FileSourceNode>();
    nlohmann::json params = {
        {"url", sampleVideoUrl()},
        {"loop", false},
        {"realtime", false}
    };
    ASSERT_TRUE(node->configure("src_direct", params));
    ASSERT_TRUE(node->start());

    ASSERT_TRUE(waitUntil([&] { return !node->isRunning(); }));

    // 直接 start，不调 stop
    ASSERT_TRUE(node->start());
    EXPECT_TRUE(node->isRunning());
    ASSERT_TRUE(waitUntil([&] { return !node->isRunning(); }));
    node->stop();
    node.reset();
}

// 运行中主动 stop：正常回收
TEST(FileSourceLifecycleTest, ActiveStopWhileRunning) {
    auto node = std::make_shared<FileSourceNode>();
    nlohmann::json params = {
        {"url", sampleVideoUrl()},
        {"loop", true},
        {"realtime", true}
    };
    ASSERT_TRUE(node->configure("src_test2", params));
    ASSERT_TRUE(node->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    EXPECT_TRUE(node->isRunning());

    node->stop();
    EXPECT_FALSE(node->isRunning());

    // 停止后可再次启动
    ASSERT_TRUE(node->start());
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    node->stop();
    node.reset();
}

// STREAM_END 广播：EOF 自停前下游应收到 STREAM_END
TEST(FileSourceLifecycleTest, StreamEndBroadcastOnEof) {
    class StreamEndSink : public core::Node {
    public:
        StreamEndSink() : core::Node("eos_sink") {}
        bool start() override { running_ = true; return true; }
        void stop() override { running_ = false; }
        void pushData(std::shared_ptr<core::BasePacket> packet) override {
            if (packet->type == core::PacketType::STREAM_END) {
                got_eos_ = true;
            }
        }
        std::atomic<bool> got_eos_{false};
    };

    auto node = std::make_shared<FileSourceNode>();
    auto sink = std::make_shared<StreamEndSink>();
    node->addDownstream(sink);
    nlohmann::json params = {
        {"url", sampleVideoUrl()},
        {"loop", false},
        {"realtime", false}
    };
    ASSERT_TRUE(node->configure("src_test3", params));
    ASSERT_TRUE(sink->start());
    ASSERT_TRUE(node->start());

    ASSERT_TRUE(waitUntil([&] { return sink->got_eos_.load(); }));
    ASSERT_TRUE(waitUntil([&] { return !node->isRunning(); }));
    node->stop();
}
