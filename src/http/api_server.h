// src/http/api_server.h
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <httplib.h>
#include "ai_stream/core/pipeline.h"

namespace ai_stream {
namespace http {

/**
 * @brief HTTP API 服务，管理多个 Pipeline 实例
 */
class ApiServer {
public:
    ApiServer();
    ~ApiServer();

    /**
     * @brief 启动 HTTP 服务器
     * @param host 监听地址
     * @param port 监听端口
     * @return 成功返回 true
     */
    bool start(const std::string& host = "0.0.0.0", int port = 8080);

    /**
     * @brief 停止服务器
     */
    void stop();

private:
    // 注册所有路由处理函数
    void setupRoutes();

    // 管道管理接口
    void handlePipelineBuild(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStart(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStop(const httplib::Request& req, httplib::Response& res);
    void handlePipelineDestroy(const httplib::Request& req, httplib::Response& res);
    void handlePipelineList(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStatus(const httplib::Request& req, httplib::Response& res);

    // 健康检查
    void handleHealth(const httplib::Request& req, httplib::Response& res);

    // 指标监控
    void handleMetricsPrometheus(const httplib::Request& req, httplib::Response& res);
    void handleMetricsJson(const httplib::Request& req, httplib::Response& res);
    void handleMetricsPipeline(const httplib::Request& req, httplib::Response& res);

private:
    httplib::Server server_;
    std::unordered_map<std::string, std::shared_ptr<core::Pipeline>> pipelines_;
    std::mutex pipelines_mutex_;
    std::thread server_thread_;
    std::atomic<bool> running_{false};
};

} // namespace http
} // namespace ai_stream