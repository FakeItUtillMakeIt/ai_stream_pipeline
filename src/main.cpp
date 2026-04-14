// src/main.cpp
#include "src/http/api_server.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <spdlog/spdlog.h>
#include <csignal>
#include <atomic>
#include <thread>

std::atomic<bool> g_running{true};

void signalHandler(int signum) {
    LOG_INFO_FMT("Interrupt signal ({}) received.", signum);
    g_running = false;
}

int main(int argc, char* argv[]) {
    // 初始化日志
    LogManager::Config config;
    config.log_dir = "./logs";
    config.log_name = "ai_stream";
    config.level = LogManager::Level::S_WARN;
    config.max_file_size = 5 * 1024 * 1024; // 5MB
    config.max_files = 5;
    config.console_output = true;
    config.async_mode = true;
    config.queue_size = 16384;
    config.flush_interval = 2;
    
    // 注册信号处理
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
    
    // 解析命令行参数（简化）
    std::string host = "0.0.0.0";
    int port = 8080;
    if (argc > 1) host = argv[1];
    if (argc > 2) port = std::stoi(argv[2]);
    
    LOG_INFO("Starting AI Stream Pipeline server...");
    
    // 启动 HTTP 服务
    ai_stream::http::ApiServer server;
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