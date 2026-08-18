// src/http/api_server.h
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <mutex>
#include <httplib.h>
#include "ai_stream/core/pipeline.h"
#include "src/core/async_pipeline_manager.h"

namespace ai_stream {
namespace http {

/**
 * @brief HTTP API 服务，管理多个 Pipeline 实例
 *
 * 支持两种管道管理模式（构造时确定）：
 * - 同步模式（默认）：build/start/stop 在请求线程内立即完成，响应即最终结果
 * - 异步模式：build/start/stop 提交到 AsyncPipelineManager 任务队列后立即返回，
 *   通过 /api/v1/pipeline/status 轮询生命周期状态（loading/running/stopped/load_failed）
 */
class ApiServer {
public:
    explicit ApiServer(bool async_mode = false);
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

    bool isAsyncMode() const { return async_mode_; }

private:
    // 注册所有路由处理函数
    void setupRoutes();

    // 管道管理接口（按模式分发）
    void handlePipelineBuild(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStart(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStop(const httplib::Request& req, httplib::Response& res);
    void handlePipelineDestroy(const httplib::Request& req, httplib::Response& res);
    void handlePipelineList(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStatus(const httplib::Request& req, httplib::Response& res);

    // 同步模式实现
    void handlePipelineBuildSync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStartSync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStopSync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineDestroySync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineListSync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStatusSync(const httplib::Request& req, httplib::Response& res);

    // 异步模式实现
    void handlePipelineBuildAsync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStartAsync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStopAsync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineDestroyAsync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineListAsync(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStatusAsync(const httplib::Request& req, httplib::Response& res);

    // 健康检查
    void handleHealth(const httplib::Request& req, httplib::Response& res);

    // 指标监控
    void handleMetricsPrometheus(const httplib::Request& req, httplib::Response& res);
    void handleMetricsJson(const httplib::Request& req, httplib::Response& res);
    void handleMetricsPipeline(const httplib::Request& req, httplib::Response& res);

private:
    httplib::Server server_;
    bool async_mode_;

    // 同步模式：直接持有管道实例
    std::unordered_map<std::string, std::shared_ptr<core::Pipeline>> pipelines_;
    std::mutex pipelines_mutex_;

    // 异步模式：任务队列管理器
    std::unique_ptr<core::AsyncPipelineManager> manager_;

    std::thread server_thread_;
    std::atomic<bool> running_{false};
};

} // namespace http
} // namespace ai_stream
