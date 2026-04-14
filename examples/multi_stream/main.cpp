#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <memory>
#include <csignal>

#include "ai_stream/core/pipeline.h"
#include <nlohmann/json.hpp>
#include "3rd_party/log_mgr/log_mgr.h"

using json = nlohmann::json;
using namespace ai_stream::core;

std::atomic<bool> g_running{true};

void signal_handler(int) {
    g_running = false;
}

// 辅助函数：根据配置模板为每个流生成特定配置
json createPipelineConfig(const std::string& id, const std::string& rtsp_url) {
    json config = {
        {"id", id},
        {"graph", {
            {"nodes", {
                {{"id", "src"}, {"type", "rtsp_source"}, {"params", {{"url", rtsp_url}}}},
                {{"id", "decode"}, {"type", "ffmpeg_decode"}},
                {{"id", "infer"}, {"type", "tensorrt_infer"}, {"params", {{"model_path", "/models/yolov8.engine"}}}},
                {{"id", "draw"}, {"type", "osd_draw"}},
                {{"id", "sink"}, {"type", "rtmp_sink"}, {"params", {{"output_url", "rtmp://localhost/live/" + id}}}}
            }},
            {"edges", {
                {{"from", "src"}, {"to", "decode"}},
                {{"from", "decode"}, {"to", "infer"}},
                {{"from", "infer"}, {"to", "draw"}},
                {{"from", "draw"}, {"to", "sink"}}
            }}
        }}
    };
    return config;
}

int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    std::signal(SIGINT, signal_handler);
    std::signal(SIGTERM, signal_handler);

    // 定义多个 RTSP 流地址（可从命令行或配置文件读取）
    std::vector<std::string> rtsp_urls = {
        "rtsp://192.168.1.101/cam1",
        "rtsp://192.168.1.102/cam2",
        "rtsp://192.168.1.103/cam3"
    };

    std::vector<std::shared_ptr<Pipeline>> pipelines;

    // 为每个流创建并启动一个管道
    for (size_t i = 0; i < rtsp_urls.size(); ++i) {
        std::string id = "stream_" + std::to_string(i+1);
        json config = createPipelineConfig(id, rtsp_urls[i]);
        
        auto pipeline = std::make_shared<Pipeline>(id);
        if (!pipeline->buildFromJson(config["graph"])) {
            LOG_ERROR_FMT("Failed to build pipeline for {}", id);
            continue;
        }
        if (!pipeline->start()) {
            LOG_ERROR_FMT("Failed to start pipeline for {}", id);
            continue;
        }
        pipelines.push_back(pipeline);
        LOG_INFO_FMT("Pipeline '{}' started for {}", id, rtsp_urls[i]);
    }

    LOG_INFO_FMT("All pipelines running. Press Ctrl+C to stop.");

    // 等待退出信号
    while (g_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }

    // 停止所有管道
    LOG_INFO_FMT("Stopping all pipelines...");
    for (auto& p : pipelines) {
        p->stop();
    }
    LOG_INFO_FMT("All pipelines stopped.");

    return 0;
}