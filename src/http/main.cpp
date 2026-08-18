// src/main.cpp
#include "src/http/api_server.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>
#include <csignal>
#include <atomic>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <thread>

std::atomic<bool> g_running{true};

void signalHandler(int signum) {
    LOG_INFO_FMT("Interrupt signal ({}) received.", signum);
    g_running = false;
}

namespace {

LogManager::Level parseLevel(const std::string& s) {
    if (s == "trace") return LogManager::Level::S_TRACE;
    if (s == "debug") return LogManager::Level::S_DEBUG;
    if (s == "info") return LogManager::Level::S_INFO;
    if (s == "warn") return LogManager::Level::S_WARN;
    if (s == "error") return LogManager::Level::S_ERROR;
    if (s == "critical") return LogManager::Level::S_CRITICAL;
    if (s == "off") return LogManager::Level::S_OFF;
    return LogManager::Level::S_INFO;
}

// 若存在 config/logging/logging.json 则用其覆盖默认日志配置
void applyLoggingConfigFile(LogManager::Config& config, const std::string& path) {
    std::ifstream f(path);
    if (!f.is_open()) return;
    try {
        nlohmann::json j = nlohmann::json::parse(f);
        if (j.contains("log_dir")) config.log_dir = j["log_dir"].get<std::string>();
        if (j.contains("log_name")) config.log_name = j["log_name"].get<std::string>();
        if (j.contains("level")) config.level = parseLevel(j["level"].get<std::string>());
        if (j.contains("max_file_size_mb")) config.max_file_size = j["max_file_size_mb"].get<size_t>() * 1024 * 1024;
        if (j.contains("max_files")) config.max_files = j["max_files"].get<size_t>();
        if (j.contains("console_output")) config.console_output = j["console_output"].get<bool>();
        if (j.contains("async_mode")) config.async_mode = j["async_mode"].get<bool>();
        if (j.contains("queue_size")) config.queue_size = j["queue_size"].get<size_t>();
        if (j.contains("flush_interval")) config.flush_interval = j["flush_interval"].get<int>();
    } catch (...) {
        // 解析失败则沿用默认配置
    }
}

} // namespace

int main(int argc, char* argv[]) {
    // 初始化日志（默认值 + 可选的 config/logging/logging.json 覆盖）
    LogManager::Config config;
    config.log_dir = "./logs";
    config.log_name = "ai_stream";
    config.level = LogManager::Level::S_INFO;
    config.max_file_size = 5 * 1024 * 1024; // 5MB
    config.max_files = 5;
    config.console_output = true;
    config.async_mode = true;
    config.queue_size = 16384;
    config.flush_interval = 2;
    applyLoggingConfigFile(config, "config/logging/logging.json");

    if (!LogManager::get_instance().initialize(config)) {
        LOG_ERROR("Failed to initialize LogManager");
    }

    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // 解析命令行参数：http_server [host] [port] [--async]
    // --async 启用异步管道管理（build/start/stop 提交任务后立即返回，状态轮询）
    std::string host = "0.0.0.0";
    int port = 8080;
    bool async_mode = false;
    std::vector<std::string> positional;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--async" || arg == "-a") {
            async_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: " << argv[0] << " [host] [port] [--async]\n"
                      << "  host     监听地址，默认 0.0.0.0\n"
                      << "  port     监听端口，默认 8080\n"
                      << "  --async  启用异步管道管理模式\n";
            return 0;
        } else {
            positional.push_back(arg);
        }
    }
    if (positional.size() > 0) host = positional[0];
    if (positional.size() > 1) port = std::stoi(positional[1]);

    LOG_INFO_FMT("Starting AI Stream Pipeline server... (mode: {})", async_mode ? "async" : "sync");

    // 启动 HTTP 服务
    ai_stream::http::ApiServer server(async_mode);
    if (!server.start(host, port)) {
        LOG_ERROR("Failed to start HTTP server");
        return 1;
    }

    LOG_INFO_FMT("Server running on {}:{}, press Ctrl+C to stop", host, port);
    
    // 主循环等待退出信号
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    LOG_INFO("Shutting down...");
    server.stop();
    LogManager::get_instance().shutdown();
    
    return 0;
}