// tests/unit/core/test_pipeline.cpp
#include <gtest/gtest.h>
#include "ai_stream/core/pipeline.h"
#include "nodes/registry/node_factory.h"
#include <algorithm>
#include <mutex>
#include <vector>

using namespace ai_stream;
using namespace ai_stream::core;

namespace {

// 记录全局启停顺序，用于验证拓扑序
class OrderRecorder {
public:
    static OrderRecorder& instance() {
        static OrderRecorder r;
        return r;
    }
    void record(const std::string& event) {
        std::lock_guard<std::mutex> lock(m_);
        events_.push_back(event);
    }
    std::vector<std::string> events() {
        std::lock_guard<std::mutex> lock(m_);
        return events_;
    }
    void reset() {
        std::lock_guard<std::mutex> lock(m_);
        events_.clear();
    }
private:
    std::mutex m_;
    std::vector<std::string> events_;
};

class DummyNode : public Node {
public:
    DummyNode() : Node("DummyNode") {}

    bool configure(const std::string& node_id, const nlohmann::json& params) override {
        id_ = node_id;
        if (params.value("fail_configure", false)) {
            return false;
        }
        return true;
    }

    bool start() override {
        running_ = true;
        OrderRecorder::instance().record("start:" + id_);
        return true;
    }
    void stop() override {
        running_ = false;
        OrderRecorder::instance().record("stop:" + id_);
    }
    void pushData(std::shared_ptr<BasePacket>) override {}

private:
    std::string id_;
};

class FailingStartNode : public Node {
public:
    FailingStartNode() : Node("FailingStartNode") {}
    bool configure(const std::string& node_id, const nlohmann::json&) override {
        id_ = node_id;
        return true;
    }
    bool start() override {
        OrderRecorder::instance().record("start:" + id_);
        return false;
    }
    void stop() override {
        OrderRecorder::instance().record("stop:" + id_);
    }
    void pushData(std::shared_ptr<BasePacket>) override {}
private:
    std::string id_;
};

} // namespace

REGISTER_NODE("test_dummy", DummyNode)
REGISTER_NODE("test_failing_start", FailingStartNode)

namespace {

nlohmann::json makeNode(const std::string& id, const std::string& type,
                        const nlohmann::json& params = nlohmann::json::object()) {
    return {{"id", id}, {"type", type}, {"params", params}};
}

nlohmann::json makeEdge(const std::string& from, const std::string& to) {
    return {{"from", from}, {"to", to}};
}

class PipelineTest : public ::testing::Test {
protected:
    void SetUp() override { OrderRecorder::instance().reset(); }
};

} // namespace

TEST_F(PipelineTest, BuildLinearChainAndVerifyTopologicalOrder) {
    // 拓扑：a -> b -> c
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy"), makeNode("b", "test_dummy"), makeNode("c", "test_dummy")}},
        {"edges", {makeEdge("a", "b"), makeEdge("b", "c")}}
    };
    auto pipeline = std::make_shared<Pipeline>("topo_test");
    ASSERT_TRUE(pipeline->buildFromJson(config));
    ASSERT_TRUE(pipeline->start());

    // 启动按逆拓扑序：下游先启动（c, b, a）
    auto events = OrderRecorder::instance().events();
    ASSERT_EQ(events.size(), 3u);
    EXPECT_EQ(events[0], "start:c");
    EXPECT_EQ(events[1], "start:b");
    EXPECT_EQ(events[2], "start:a");

    OrderRecorder::instance().reset();
    pipeline->stop();

    // 停止按拓扑序：上游先停（a, b, c）
    events = OrderRecorder::instance().events();
    ASSERT_EQ(events.size(), 3u);
    EXPECT_EQ(events[0], "stop:a");
    EXPECT_EQ(events[1], "stop:b");
    EXPECT_EQ(events[2], "stop:c");
}

TEST_F(PipelineTest, BuildDiamondGraph) {
    // a -> b, a -> c, b -> d, c -> d
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy"), makeNode("b", "test_dummy"),
                   makeNode("c", "test_dummy"), makeNode("d", "test_dummy")}},
        {"edges", {makeEdge("a", "b"), makeEdge("a", "c"), makeEdge("b", "d"), makeEdge("c", "d")}}
    };
    auto pipeline = std::make_shared<Pipeline>("diamond_test");
    ASSERT_TRUE(pipeline->buildFromJson(config));
    ASSERT_TRUE(pipeline->start());

    auto events = OrderRecorder::instance().events();
    ASSERT_EQ(events.size(), 4u);
    // 逆拓扑序约束：d 必须在 b/c 之前，b/c 必须在 a 之前
    auto pos = [&](const std::string& name) {
        return std::find(events.begin(), events.end(), name) - events.begin();
    };
    EXPECT_LT(pos("start:d"), pos("start:b"));
    EXPECT_LT(pos("start:d"), pos("start:c"));
    EXPECT_LT(pos("start:b"), pos("start:a"));
    EXPECT_LT(pos("start:c"), pos("start:a"));
    pipeline->stop();
}

TEST_F(PipelineTest, CycleDetectionFails) {
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy"), makeNode("b", "test_dummy")}},
        {"edges", {makeEdge("a", "b"), makeEdge("b", "a")}}
    };
    auto pipeline = std::make_shared<Pipeline>("cycle_test");
    EXPECT_FALSE(pipeline->buildFromJson(config));
}

TEST_F(PipelineTest, SelfLoopFails) {
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy")}},
        {"edges", {makeEdge("a", "a")}}
    };
    auto pipeline = std::make_shared<Pipeline>("selfloop_test");
    EXPECT_FALSE(pipeline->buildFromJson(config));
}

TEST_F(PipelineTest, InvalidEdgeFails) {
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy")}},
        {"edges", {makeEdge("a", "ghost")}}
    };
    auto pipeline = std::make_shared<Pipeline>("bad_edge_test");
    EXPECT_FALSE(pipeline->buildFromJson(config));
}

TEST_F(PipelineTest, DuplicateNodeIdFails) {
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy"), makeNode("a", "test_dummy")}},
    };
    auto pipeline = std::make_shared<Pipeline>("dup_id_test");
    EXPECT_FALSE(pipeline->buildFromJson(config));
}

TEST_F(PipelineTest, MissingNodesArrayFails) {
    auto pipeline = std::make_shared<Pipeline>("no_nodes_test");
    EXPECT_FALSE(pipeline->buildFromJson(nlohmann::json::object()));
}

TEST_F(PipelineTest, UnknownNodeTypeFails) {
    nlohmann::json config = {
        {"nodes", {makeNode("a", "no_such_type")}},
    };
    auto pipeline = std::make_shared<Pipeline>("unknown_type_test");
    EXPECT_FALSE(pipeline->buildFromJson(config));
}

TEST_F(PipelineTest, ConfigureFailureAbortsBuild) {
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy", {{"fail_configure", true}})}},
    };
    auto pipeline = std::make_shared<Pipeline>("cfg_fail_test");
    EXPECT_FALSE(pipeline->buildFromJson(config));
}

TEST_F(PipelineTest, StartFailureRollsBackStartedNodes) {
    // a -> b，b 启动失败：已启动的 a 必须被回滚停止
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy"), makeNode("b", "test_failing_start")}},
        {"edges", {makeEdge("a", "b")}}
    };
    auto pipeline = std::make_shared<Pipeline>("rollback_test");
    ASSERT_TRUE(pipeline->buildFromJson(config));
    EXPECT_FALSE(pipeline->start());

    auto events = OrderRecorder::instance().events();
    // 逆拓扑启动：b 先启动并失败
    ASSERT_GE(events.size(), 1u);
    EXPECT_EQ(events[0], "start:b");
    // 回滚：a 被停止
    bool a_stopped = std::find(events.begin(), events.end(), "stop:a") != events.end();
    EXPECT_TRUE(a_stopped);
    EXPECT_FALSE(pipeline->isRunning());
}

TEST_F(PipelineTest, GetNodeReturnsBuiltNode) {
    nlohmann::json config = {
        {"nodes", {makeNode("a", "test_dummy")}},
    };
    auto pipeline = std::make_shared<Pipeline>("getnode_test");
    ASSERT_TRUE(pipeline->buildFromJson(config));
    EXPECT_NE(pipeline->getNode("a"), nullptr);
    EXPECT_EQ(pipeline->getNode("ghost"), nullptr);
}
