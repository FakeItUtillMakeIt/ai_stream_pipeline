#include <iostream>
#include <csignal>
#include <fstream>

#include "ai_stream/core/pipeline.h"
#include <nlohmann/json.hpp>
#include "3rd_party/log_mgr/log_mgr.h"

// 包含自定义节点头文件以触发静态注册（实际注册发生在 .cpp 编译单元）
#include "my_custom_node.h"

using json = nlohmann::json;
using namespace ai_stream::core;

std::atomic<bool> g_running{true};

void signal_handler(int) { g_running = false; }

int main() {
    spdlog::set_level(spdlog::level::debug);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // 创建一个简单的管道，包含自定义的灰度节点
    json config = {
        {"id", "custom_demo"},
        {"graph", {
            {"nodes", {
                {{"id", "src"}, {"type", "mock_source"}},
                {{"id", "gray"}, {"type", "grayscale"}},
                {{"id", "sink"}, {"type", "mock_sink"}}  // 需要确保 mock_sink 已在测试库中注册
            }},
            {"edges", {
                {{"from", "src"}, {"to", "gray"}},
                {{"from", "gray"}, {"to", "sink"}}
            }}
        }}
    };

    Pipeline pipeline("custom");
    if (!pipeline.buildFromJson(config["graph"])) {
        LOG_ERROR_FMT("Failed to build pipeline");
        return 1;
    }
    
    pipeline.start();
    LOG_INFO_FMT("Custom pipeline running. Press Ctrl+C to stop.");

    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    pipeline.stop();
    return 0;
}