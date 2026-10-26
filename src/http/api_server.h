// src/http/api_server.h
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <mutex>
#include <httplib.h>
#include "ai_stream/core/pipeline.h"
#include "src/core/async_pipeline_manager.h"
#include "request_log.h"

namespace ai_stream {
namespace http {

/**
 * @brief 每个管道的运行时元数据（供预览页取用）
 *  - play_url:    外部媒体服务器(mediamtx/nginx-rtmp)转出的 HLS/WebRTC 播放地址，取自 sink 节点 params.play_url
 *  - evidence_dirs: 证据目录（取自 evidence 节点 video/snapshot 的 output_dir，相对程序运行目录解析）
 */
struct PipelineMeta {
    std::string play_url;
    std::vector<std::string> evidence_dirs;
};

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

    // 解析请求体 JSON；失败时置 400 并返回 false（区分解析错误与内部错误）
    bool parseJsonBody(const httplib::Request& req, httplib::Response& res, nlohmann::json& out);

    // 管道管理接口（按模式分发）
    void handlePipelineBuild(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStart(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStop(const httplib::Request& req, httplib::Response& res);
    void handlePipelineDestroy(const httplib::Request& req, httplib::Response& res);
    void handlePipelineList(const httplib::Request& req, httplib::Response& res);
    void handlePipelineStatus(const httplib::Request& req, httplib::Response& res);
    // 保存管道配置文件到 <state>/pipelines/<id>.json（编辑器"保存到服务器"）
    void handleSaveConfig(const httplib::Request& req, httplib::Response& res);
    // 读取某管道已保存的配置（编辑器"从服务器导入/同步"）
    void handleGetConfig(const httplib::Request& req, httplib::Response& res);
    // 证据查看（预览页）：列目录 / 取文件（限制在该管道 evidence 目录内）
    void handleEvidenceList(const httplib::Request& req, httplib::Response& res);
    void handleEvidenceFile(const httplib::Request& req, httplib::Response& res);
    // 从 build 请求体抽取 play_url / evidence 目录，登记到 meta_
    void extractMeta(const std::string& pipeline_id, const nlohmann::json& body);

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

    // ===== 请求持久化 / 掉线重启恢复 =====
    // 记录一次平台请求。build 会把请求体存成配置文件，其余动作复用同一路径。
    void recordRequest(const std::string& action, const std::string& pipeline_id,
                       const nlohmann::json* build_body = nullptr);
    // 启动时读取历史最后一行：仅当是 start 才据此重建并启动，否则等待。
    void recoverFromState();

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

    // 请求历史持久化（掉线/重启恢复）
    std::unique_ptr<RequestLog> reqlog_;

    // 每管道运行时元数据（play_url / evidence 目录），供预览页
    std::map<std::string, PipelineMeta> meta_;
    std::mutex meta_mutex_;

    std::thread server_thread_;
    std::atomic<bool> running_{false};

    // 可选 bearer token 鉴权（环境变量 AI_STREAM_API_TOKEN，未设置则不启用）
    std::string api_token_;
};

} // namespace http
} // namespace ai_stream
