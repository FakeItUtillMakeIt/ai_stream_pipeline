#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <fstream>

#include "3rd_party/log_mgr/log_mgr.h"
#include "ai_stream/core/pipeline.h"
#include "ai_stream/nodes/i_source_node.h"
#include "ai_stream/nodes/i_infer_node.h"
#include <nlohmann/json.hpp>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

using json = nlohmann::json;
using namespace ai_stream::core;

std::atomic<bool> g_running{true};

void signal_handler(int)
{
    g_running = false;
}

int main(int argc, char *argv[])
{
    // 设置日志
    LogManager::Config log_config;
    log_config.log_dir = "./logs";
    log_config.log_name = "simple_detection";
    log_config.level = LogManager::Level::S_DEBUG;
    log_config.max_file_size = 5 * 1024 * 1024; // 5MB
    log_config.max_files = 5;
    log_config.console_output = true;
    log_config.async_mode = false;
    log_config.queue_size = 16384;
    log_config.flush_interval = 2;

    if (!LogManager::get_instance().initialize(log_config))
    {
        std::cerr << "Failed to initialize logger!" << std::endl;
        return -1;
    }

    // 处理退出信号
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // 检查命令行参数
    if (argc < 2)
    {
        LOG_ERROR_FMT("Usage: {} <pipeline_config.json>", argv[0]);

        return 1;
    }
    LOG_INFO("-------------------Start simple detection------------------");
    // 读取管道配置文件
    std::ifstream config_file(argv[1]);
    if (!config_file.is_open())
    {
        LOG_ERROR_FMT("Cannot open config file: {}", argv[1]);
        return 1;
    }
    json config = json::parse(config_file);

    // 创建管道并构建
    std::string pipeline_id = config.value("id", "example_pipeline");
    auto pipeline = std::make_shared<Pipeline>(pipeline_id);

    if (!pipeline->buildFromJson(config["graph"]))
    {
        LOG_ERROR_FMT("Failed to build pipeline from config");
        return 1;
    }

    // 启动管道
    if (!pipeline->start())
    {
        LOG_ERROR_FMT("Failed to start pipeline");
        LOG_INFO("Stopping pipeline...");
        pipeline->stop();
        LOG_INFO("Pipeline stopped. Exiting.");
        LogManager::get_instance().shutdown();
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        return 1;
    }

    LOG_INFO_FMT("Pipeline '{}' is running. Press Ctrl+C to stop.", pipeline_id);

    // 主循环等待退出信号
    while (g_running)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    LOG_INFO("Stopping pipeline...");
    pipeline->stop();
    LOG_INFO("Pipeline stopped. Exiting.");
    LogManager::get_instance().shutdown();

    return 0;
}