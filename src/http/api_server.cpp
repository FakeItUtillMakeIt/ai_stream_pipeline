// src/http/api_server.cpp
#include "api_server.h"
#include "ai_stream/core/metrics.h"
#include "ai_stream/hal/backend_diagnostics.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>

using json = nlohmann::json;

namespace ai_stream {
namespace http {

namespace {
// 安全提取 id：缺失/非字符串均返回空串，避免 json::type_error 变成 500
std::string idOf(const json& body) {
    if (body.is_object() && body.contains("id") && body["id"].is_string()) {
        return body["id"].get<std::string>();
    }
    return "";
}
} // namespace

ApiServer::ApiServer(bool async_mode) : async_mode_(async_mode) {
    if (async_mode_) {
        manager_ = std::make_unique<core::AsyncPipelineManager>();
        LOG_INFO_FMT("ApiServer pipeline management mode: ASYNC");
    } else {
        LOG_INFO_FMT("ApiServer pipeline management mode: SYNC");
    }
    if (const char* token = std::getenv("AI_STREAM_API_TOKEN")) {
        api_token_ = token;
        if (!api_token_.empty()) {
            LOG_WARN("ApiServer API token auth enabled (AI_STREAM_API_TOKEN)");
        }
    }
    hal::logAvailableBackends();
    setupRoutes();
}

bool ApiServer::parseJsonBody(const httplib::Request& req, httplib::Response& res, json& out) {
    try {
        out = json::parse(req.body);
        return true;
    } catch (const json::parse_error& e) {
        res.status = 400;
        res.set_content(json{{"error", std::string("Invalid JSON: ") + e.what()}}.dump(),
                        "application/json");
        return false;
    }
}

ApiServer::~ApiServer() {
    stop();
}

void ApiServer::setupRoutes() {
    // 请求体上限与读写超时（防超大 body / 慢连接拖垮服务）
    server_.set_payload_max_length(16 * 1024 * 1024);  // 16 MB
    server_.set_read_timeout(10, 0);
    server_.set_write_timeout(60, 0);

    // CORS 支持（开发环境）+ 可选 token 鉴权
    server_.set_pre_routing_handler([this](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type, Authorization");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
        }
        // 仅保护 /api/ 业务接口；/health、/metrics 放行
        if (!api_token_.empty() && req.path.rfind("/api/", 0) == 0) {
            if (req.get_header_value("Authorization") != ("Bearer " + api_token_)) {
                res.status = 401;
                res.set_content(R"({"error":"Unauthorized"})", "application/json");
                return httplib::Server::HandlerResponse::Handled;
            }
        }
        return httplib::Server::HandlerResponse::Unhandled;
    });

    // 管道管理 API
    server_.Post("/api/v1/pipeline/build", [this](const auto& req, auto& res) {
        handlePipelineBuild(req, res);
    });
    server_.Post("/api/v1/pipeline/start", [this](const auto& req, auto& res) {
        handlePipelineStart(req, res);
    });
    server_.Post("/api/v1/pipeline/stop", [this](const auto& req, auto& res) {
        handlePipelineStop(req, res);
    });
    server_.Delete("/api/v1/pipeline/delete", [this](const auto& req, auto& res) {
        handlePipelineDestroy(req, res);
    });
    server_.Get("/api/v1/pipeline/list", [this](const auto& req, auto& res) {
        handlePipelineList(req, res);
    });
    server_.Post("/api/v1/pipeline/status", [this](const auto& req, auto& res) {
        handlePipelineStatus(req, res);
    });

    // 健康检查
    server_.Get("/health", [this](const auto& req, auto& res) {
        handleHealth(req, res);
    });

    // 指标监控
    server_.Get("/metrics", [this](const auto& req, auto& res) {
        handleMetricsPrometheus(req, res);
    });
    server_.Get("/api/v1/metrics", [this](const auto& req, auto& res) {
        handleMetricsJson(req, res);
    });
    server_.Post("/api/v1/metrics", [this](const auto& req, auto& res) {
        handleMetricsPipeline(req, res);
    });

    // 可用后端查询
    server_.Get("/api/v1/backends", [](const auto& req, auto& res) {
        (void)req;
        try {
            res.set_content(ai_stream::hal::availableBackendsJson(), "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            json err = {{"error", e.what()}};
            res.set_content(err.dump(), "application/json");
        }
    });
}

bool ApiServer::start(const std::string& host, int port) {
    if (running_.exchange(true)) return true;

    server_thread_ = std::thread([this, host, port]() {
        LOG_INFO_FMT("HTTP server listening on {}:{}", host, port);
        server_.listen(host.c_str(), port);
        running_ = false;
    });

    // 轮询等待服务就绪或线程退出，替代固定 sleep（避免慢机器误判与竞态）
    for (int i = 0; i < 100 && running_.load(); ++i) {
        if (server_.is_running()) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    return server_.is_running();
}

void ApiServer::stop() {
    // 无论 running_ 状态如何都必须 join 线程（listen 失败/启动中途停止时
    // running_ 可能为 false，但 server_thread_ 仍 joinable，否则析构触发 terminate）
    running_ = false;
    server_.stop();
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    // 停止所有管道（异步模式由 manager 析构统一停止）
    if (!async_mode_) {
        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        for (auto& [id, pipeline] : pipelines_) {
            pipeline->stop();
        }
        pipelines_.clear();
    } else {
        manager_.reset();
    }
    LOG_INFO_FMT("HTTP server stopped");
}

// ==============================================================================
// 模式分发
// ==============================================================================

void ApiServer::handlePipelineBuild(const httplib::Request& req, httplib::Response& res) {
    if (async_mode_) handlePipelineBuildAsync(req, res);
    else handlePipelineBuildSync(req, res);
}

void ApiServer::handlePipelineStart(const httplib::Request& req, httplib::Response& res) {
    if (async_mode_) handlePipelineStartAsync(req, res);
    else handlePipelineStartSync(req, res);
}

void ApiServer::handlePipelineStop(const httplib::Request& req, httplib::Response& res) {
    if (async_mode_) handlePipelineStopAsync(req, res);
    else handlePipelineStopSync(req, res);
}

void ApiServer::handlePipelineDestroy(const httplib::Request& req, httplib::Response& res) {
    if (async_mode_) handlePipelineDestroyAsync(req, res);
    else handlePipelineDestroySync(req, res);
}

void ApiServer::handlePipelineList(const httplib::Request& req, httplib::Response& res) {
    if (async_mode_) handlePipelineListAsync(req, res);
    else handlePipelineListSync(req, res);
}

void ApiServer::handlePipelineStatus(const httplib::Request& req, httplib::Response& res) {
    if (async_mode_) handlePipelineStatusAsync(req, res);
    else handlePipelineStatusSync(req, res);
}

// ==============================================================================
// 同步模式实现
// ==============================================================================

void ApiServer::handlePipelineBuildSync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);
        if (pipeline_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"Missing pipeline id"})", "application/json");
            return;
        }
        if (!body.contains("graph")) {
            res.status = 400;
            res.set_content(R"({"error":"Missing graph configuration"})", "application/json");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(pipelines_mutex_);
            if (pipelines_.find(pipeline_id) != pipelines_.end()) {
                res.status = 409;
                res.set_content(R"({"error":"Pipeline id already exists"})", "application/json");
                return;
            }
        }

        // 构建（含模型加载，可能耗时）放在全局锁外，避免阻塞其它请求
        auto pipeline = std::make_shared<core::Pipeline>(pipeline_id);
        if (!pipeline->buildFromJson(body["graph"])) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid pipeline configuration"})", "application/json");
            return;
        }

        {
            std::lock_guard<std::mutex> lock(pipelines_mutex_);
            if (pipelines_.find(pipeline_id) != pipelines_.end()) {
                res.status = 409;
                res.set_content(R"({"error":"Pipeline id already exists"})", "application/json");
                return;
            }
            pipelines_[pipeline_id] = pipeline;
        }

        json response = {
            {"status", "ok"},
            {"id", pipeline_id},
            {"message", "Pipeline built successfully"}
        };
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' built via API", pipeline_id);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void ApiServer::handlePipelineStartSync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);

        std::shared_ptr<core::Pipeline> pipeline;
        {
            std::lock_guard<std::mutex> lock(pipelines_mutex_);
            auto it = pipelines_.find(pipeline_id);
            if (it == pipelines_.end()) {
                res.status = 404;
                res.set_content(R"({"error":"Pipeline not found"})", "application/json");
                return;
            }
            pipeline = it->second;
        }

        // 启动（可能涉及线程/资源）放在锁外
        if (!pipeline->start()) {
            res.status = 500;
            res.set_content(R"({"error":"Failed to start pipeline"})", "application/json");
            return;
        }

        json response = {{"status", "ok"}, {"id", pipeline_id}, {"running", true}};
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' started via API", pipeline_id);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void ApiServer::handlePipelineStopSync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);

        std::shared_ptr<core::Pipeline> pipeline;
        {
            std::lock_guard<std::mutex> lock(pipelines_mutex_);
            auto it = pipelines_.find(pipeline_id);
            if (it == pipelines_.end()) {
                res.status = 404;
                res.set_content(R"({"error":"Pipeline not found"})", "application/json");
                return;
            }
            pipeline = it->second;
        }

        pipeline->stop();
        json response = {{"status", "ok"}, {"id", pipeline_id}, {"running", false}};
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' stopped via API", pipeline_id);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void ApiServer::handlePipelineDestroySync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);

        std::shared_ptr<core::Pipeline> pipeline;
        {
            std::lock_guard<std::mutex> lock(pipelines_mutex_);
            auto it = pipelines_.find(pipeline_id);
            if (it == pipelines_.end()) {
                res.status = 404;
                res.set_content(R"({"error":"Pipeline not found"})", "application/json");
                return;
            }
            pipeline = it->second;
            pipelines_.erase(it);
        }

        if (pipeline) pipeline->stop();
        // 清理该管道的指标，避免 metrics_ 无界增长
        core::MetricsCollector::instance().reset(pipeline_id);

        json response = {{"status", "ok"}, {"id", pipeline_id}, {"message", "Pipeline destroyed"}};
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' destroyed via API", pipeline_id);
    }
    catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

void ApiServer::handlePipelineListSync(const httplib::Request& /*req*/, httplib::Response& res) {
    try {
        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        json list = json::array();
        for (const auto& [id, pipeline] : pipelines_) {
            list.push_back({
                {"id", id},
                {"running", pipeline->isRunning()}
            });
        }
        json response = {{"pipelines", list}};
        res.set_content(response.dump(), "application/json");
    }
    catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

void ApiServer::handlePipelineStatusSync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);

        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        auto it = pipelines_.find(pipeline_id);
        if (it == pipelines_.end()) {
            res.status = 404;
            json response = {{"status", "error"}, {"message", "Pipeline not found"}, {"id", pipeline_id}};
            res.set_content(response.dump(), "application/json");
            return;
        }

        json response = {
            {"id", pipeline_id},
            {"running", it->second->isRunning()}
        };
        res.set_content(response.dump(), "application/json");
    }
    catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

// ==============================================================================
// 异步模式实现（任务提交后立即返回，状态通过 status 接口轮询）
// ==============================================================================

void ApiServer::handlePipelineBuildAsync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        LOG_INFO_FMT("Received async pipeline build request: {}", idOf(body));

        std::string pipeline_id = manager_->loadPipelineFromJsonAsync(body);
        if (pipeline_id.empty()) {
            res.status = manager_->hasPipeline(idOf(body)) ? 409 : 400;
            json err = {{"error", manager_->hasPipeline(idOf(body))
                                     ? "Pipeline id already exists"
                                     : "Invalid pipeline configuration (missing id)"}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        res.status = 202;
        json response = {
            {"status", "accepted"},
            {"id", pipeline_id},
            {"state", "loading"},
            {"message", "Pipeline build submitted, poll /api/v1/pipeline/status for progress"}
        };
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' build submitted via API (async)", pipeline_id);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void ApiServer::handlePipelineStartAsync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);

        if (!manager_->startPipelineAsync(pipeline_id)) {
            res.status = 404;
            res.set_content(R"({"error":"Pipeline not found"})", "application/json");
            return;
        }

        res.status = 202;
        json response = {{"status", "accepted"}, {"id", pipeline_id}, {"state", "starting"}};
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' start submitted via API (async)", pipeline_id);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void ApiServer::handlePipelineStopAsync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);

        if (!manager_->stopPipelineAsync(pipeline_id)) {
            res.status = 404;
            res.set_content(R"({"error":"Pipeline not found"})", "application/json");
            return;
        }

        res.status = 202;
        json response = {{"status", "accepted"}, {"id", pipeline_id}, {"state", "stopping"}};
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' stop submitted via API (async)", pipeline_id);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void ApiServer::handlePipelineDestroyAsync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);

        if (!manager_->removePipeline(pipeline_id)) {
            res.status = 404;
            res.set_content(R"({"error":"Pipeline not found"})", "application/json");
            return;
        }

        res.status = 202;
        json response = {{"status", "accepted"}, {"id", pipeline_id}, {"message", "Pipeline destroy submitted"}};
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' destroy submitted via API (async)", pipeline_id);
    }
    catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

void ApiServer::handlePipelineListAsync(const httplib::Request& /*req*/, httplib::Response& res) {
    try {
        json list = json::array();
        for (const auto& id : manager_->getAllPipelineIds()) {
            auto state = manager_->getPipelineLifecycleState(id);
            list.push_back({
                {"id", id},
                {"state", core::AsyncPipelineManager::stateToString(state)},
                {"running", manager_->getPipelineState(id)}
            });
        }
        json response = {{"pipelines", list}};
        res.set_content(response.dump(), "application/json");
    }
    catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

void ApiServer::handlePipelineStatusAsync(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);

        auto state = manager_->getPipelineLifecycleState(pipeline_id);
        if (state == core::AsyncPipelineManager::PipelineState::UNKNOWN) {
            res.status = 404;
            json response = {{"status", "error"}, {"message", "Pipeline not found"}, {"id", pipeline_id}};
            res.set_content(response.dump(), "application/json");
            return;
        }

        json response = {
            {"id", pipeline_id},
            {"state", core::AsyncPipelineManager::stateToString(state)},
            {"running", manager_->getPipelineState(pipeline_id)}
        };
        res.set_content(response.dump(), "application/json");
    }
    catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

// ==============================================================================
// 健康检查与指标
// ==============================================================================

void ApiServer::handleHealth(const httplib::Request& /*req*/, httplib::Response& res) {
    try {
        json health = {
            {"status", "healthy"},
            {"mode", async_mode_ ? "async" : "sync"},
            {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
                std::chrono::system_clock::now().time_since_epoch()).count()}
        };
        res.set_content(health.dump(), "application/json");
    }
    catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

void ApiServer::handleMetricsPrometheus(const httplib::Request& /*req*/, httplib::Response& res) {
    auto& mc = core::MetricsCollector::instance();

    // 同步模式下手动更新管道计数；异步模式由 AsyncPipelineManager 监控线程更新
    if (!async_mode_) {
        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        int active = 0;
        for (const auto& [_, p] : pipelines_) {
            if (p && p->isRunning()) active++;
        }
        mc.setActivePipelines(active);
        mc.setTotalPipelines(static_cast<int>(pipelines_.size()));
    }

    res.set_content(mc.formatPrometheus(), "text/plain; charset=utf-8");
}

void ApiServer::handleMetricsJson(const httplib::Request& /*req*/, httplib::Response& res) {
    auto& mc = core::MetricsCollector::instance();

    if (!async_mode_) {
        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        int active = 0;
        for (const auto& [_, p] : pipelines_) {
            if (p && p->isRunning()) active++;
        }
        mc.setActivePipelines(active);
        mc.setTotalPipelines(static_cast<int>(pipelines_.size()));
    }

    res.set_content(mc.formatJson(), "application/json");
}

void ApiServer::handleMetricsPipeline(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try{
        std::string pipeline_id = idOf(body);

        auto& mc = core::MetricsCollector::instance();
        auto nodes = mc.getPipelineMetrics(pipeline_id);

        if (nodes.empty()) {
            res.status = 404;
            json err = {{"error", "No metrics found for pipeline"}, {"pipeline_id", pipeline_id}};
            res.set_content(err.dump(), "application/json");
            return;
        }

        json nodes_json = json::array();
        for (const auto& m : nodes) {
            uint64_t avg = m.total_packets > 0 ? (m.total_latency_ms / m.total_packets) : 0;
            nodes_json.push_back({
                {"node_name", m.node_name},
                {"total_packets", m.total_packets},
                {"dropped_packets", m.dropped_packets},
                {"latency_ms", {
                    {"avg", avg},
                    {"min", m.min_latency_ms},
                    {"max", m.max_latency_ms},
                    {"last", m.last_latency_ms}
                }},
                {"fps", m.fps}
            });
        }

        json response = {
            {"pipeline_id", pipeline_id},
            {"nodes", nodes_json}
        };
        res.set_content(response.dump(2), "application/json");
    }
    catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

} // namespace http
} // namespace ai_stream
