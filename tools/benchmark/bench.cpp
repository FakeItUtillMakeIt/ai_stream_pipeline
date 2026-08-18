// tools/benchmark/bench.cpp
// 管道性能基准测试：加载管道配置，运行指定时长，输出各节点吞吐/延迟指标
#include "ai_stream/core/pipeline.h"
#include "ai_stream/core/metrics.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <csignal>
#include <fstream>
#include <iostream>
#include <atomic>
#include <thread>

namespace {
std::atomic<bool> g_running{true};
void signalHandler(int) { g_running = false; }
} // namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <pipeline_config.json> [duration_sec=30]\n";
        return 1;
    }

    std::string config_path = argv[1];
    int duration_sec = argc > 2 ? std::atoi(argv[2]) : 30;

    LogManager::Config log_config;
    log_config.log_dir = "./logs";
    log_config.log_name = "bench";
    log_config.level = LogManager::Level::S_WARN;
    log_config.console_output = true;
    log_config.async_mode = false;
    LogManager::get_instance().initialize(log_config);

    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::ifstream f(config_path);
    if (!f.is_open()) {
        std::cerr << "Failed to open config: " << config_path << "\n";
        return 1;
    }

    nlohmann::json config;
    try {
        config = nlohmann::json::parse(f);
    } catch (const std::exception& e) {
        std::cerr << "Failed to parse config: " << e.what() << "\n";
        return 1;
    }

    if (!config.contains("graph")) {
        std::cerr << "Config missing 'graph' section\n";
        return 1;
    }

    std::string pipeline_id = config.value("id", "bench_pipeline");
    auto pipeline = std::make_shared<ai_stream::core::Pipeline>(pipeline_id);
    if (!pipeline->buildFromJson(config["graph"])) {
        std::cerr << "Failed to build pipeline\n";
        return 1;
    }

    if (!pipeline->start()) {
        std::cerr << "Failed to start pipeline\n";
        return 1;
    }

    std::cout << "Pipeline '" << pipeline_id << "' running for " << duration_sec << "s...\n";
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(duration_sec);
    while (g_running && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    pipeline->stop();

    auto& metrics = ai_stream::core::MetricsCollector::instance();
    std::cout << "\n===== Benchmark Results =====\n";
    std::cout << metrics.formatJson() << "\n";

    LogManager::get_instance().shutdown();
    return 0;
}
