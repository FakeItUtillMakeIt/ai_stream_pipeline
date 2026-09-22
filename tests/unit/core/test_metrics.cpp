// tests/unit/core/test_metrics.cpp
#include <gtest/gtest.h>
#include "ai_stream/core/metrics.h"
#include <string>

using namespace ai_stream::core;

TEST(MetricsTest, HotPathRecordAndReset) {
    auto& mc = MetricsCollector::instance();
    mc.resetAll();
    mc.record("p1", "n1", 5);
    mc.record("p1", "n1", 15);
    auto m = mc.getNodeMetrics("p1", "n1");
    EXPECT_EQ(m.total_packets, 2u);
    EXPECT_EQ(m.total_latency_ms, 20u);
    EXPECT_EQ(m.min_latency_ms, 5u);
    EXPECT_EQ(m.max_latency_ms, 15u);
    EXPECT_EQ(m.last_latency_ms, 15u);

    mc.reset("p1");
    EXPECT_EQ(mc.getNodeMetrics("p1", "n1").total_packets, 0u);
}

TEST(MetricsTest, ZeroLatencyIsNotTreatedAsUnset) {
    auto& mc = MetricsCollector::instance();
    mc.resetAll();
    mc.record("p", "n", 0);
    EXPECT_EQ(mc.getNodeMetrics("p", "n").min_latency_ms, 0u);
    // Prometheus 中 min 应输出 0 而非哨兵值
    auto s = mc.formatPrometheus();
    EXPECT_NE(s.find("type=\"min\"} 0"), std::string::npos);
}

TEST(MetricsTest, AverageLatencyKeepsFraction) {
    auto& mc = MetricsCollector::instance();
    mc.resetAll();
    mc.record("p", "n", 1);
    mc.record("p", "n", 2);
    auto s = mc.formatJson();
    EXPECT_NE(s.find("1.5"), std::string::npos);
}

TEST(MetricsTest, PrometheusLabelEscaping) {
    auto& mc = MetricsCollector::instance();
    mc.resetAll();
    const std::string pid = "a\"} 1\nfake{x=\"y";
    mc.record(pid, "node\"x", 3);
    auto s = mc.formatPrometheus();
    // 未转义的裸引号/换行不应出现（防止注入伪造指标）
    EXPECT_EQ(s.find("a\"}"), std::string::npos);
    EXPECT_NE(s.find("a\\\"} 1\\nfake{x=\\\"y"), std::string::npos);
    mc.resetAll();
}

TEST(MetricsTest, DroppedCounter) {
    auto& mc = MetricsCollector::instance();
    mc.resetAll();
    mc.record("p", "n", 3);
    mc.recordDropped("p", "n");
    mc.recordDropped("p", "n");
    auto m = mc.getNodeMetrics("p", "n");
    EXPECT_EQ(m.total_packets, 1u);
    EXPECT_EQ(m.dropped_packets, 2u);
    mc.resetAll();
}
