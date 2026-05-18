// src/http/api_server.cpp
#include "api_server.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <nlohmann/json.hpp>
#include <chrono>

using json = nlohmann::json;

namespace ai_stream {
namespace http {

ApiServer::ApiServer() {
    setupRoutes();
}

ApiServer::~ApiServer() {
    stop();
}

void ApiServer::setupRoutes() {
    // CORS 支持（开发环境）
    server_.set_pre_routing_handler([](const httplib::Request& req, httplib::Response& res) {
        res.set_header("Access-Control-Allow-Origin", "*");
        res.set_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS");
        res.set_header("Access-Control-Allow-Headers", "Content-Type");
        if (req.method == "OPTIONS") {
            res.status = 204;
            return httplib::Server::HandlerResponse::Handled;
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
    server_.Delete("/api/v1/pipeline/:id", [this](const auto& req, auto& res) {
        handlePipelineDestroy(req, res);
    });
    server_.Get("/api/v1/pipeline/list", [this](const auto& req, auto& res) {
        handlePipelineList(req, res);
    });
    server_.Get("/api/v1/pipeline/:id/status", [this](const auto& req, auto& res) {
        handlePipelineStatus(req, res);
    });

    // 健康检查
    server_.Get("/health", [this](const auto& req, auto& res) {
        handleHealth(req, res);
    });
}

bool ApiServer::start(const std::string& host, int port) {
    if (running_) return true;

    server_thread_ = std::thread([this, host, port]() {
        running_ = true;
        LOG_INFO_FMT("HTTP server listening on {}:{}", host, port);
        server_.listen(host.c_str(), port);
        running_ = false;
    });

    // 等待一小段时间确保服务启动
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    return server_.is_running();
}

void ApiServer::stop() {
    if (!running_) return;
    server_.stop();
    if (server_thread_.joinable()) {
        server_thread_.join();
    }
    // 停止所有管道
    {
        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        for (auto& [id, pipeline] : pipelines_) {
            pipeline->stop();
        }
        pipelines_.clear();
    }
    LOG_INFO_FMT("HTTP server stopped");
}

void ApiServer::handlePipelineBuild(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        LOG_INFO_FMT("Received pipeline request:{}",body.dump());
        std::string pipeline_id = body.value("id", "");
        if (pipeline_id.empty()) {
            res.status = 400;
            res.set_content(R"({"error":"Missing pipeline id"})", "application/json");
            return;
        }

        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        if (pipelines_.find(pipeline_id) != pipelines_.end()) {
            res.status = 409;
            res.set_content(R"({"error":"Pipeline id already exists"})", "application/json");
            return;
        }

        auto pipeline = std::make_shared<core::Pipeline>(pipeline_id);
        if (!body.contains("graph")) {
            res.status = 400;
            res.set_content(R"({"error":"Missing graph configuration"})", "application/json");
            return;
        }

        if (!pipeline->buildFromJson(body["graph"])) {
            res.status = 400;
            res.set_content(R"({"error":"Invalid pipeline configuration"})", "application/json");
            return;
        }

        pipelines_[pipeline_id] = pipeline;
        
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

void ApiServer::handlePipelineStart(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        LOG_INFO_FMT("Received pipeline request:{}",body.dump());
        std::string pipeline_id = body.value("id", "");
        
        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        auto it = pipelines_.find(pipeline_id);
        if (it == pipelines_.end()) {
            res.status = 404;
            res.set_content(R"({"error":"Pipeline not found"})", "application/json");
            return;
        }

        if (!it->second->start()) {
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

void ApiServer::handlePipelineStop(const httplib::Request& req, httplib::Response& res) {
    try {
        json body = json::parse(req.body);
        std::string pipeline_id = body.value("id", "");
        
        std::lock_guard<std::mutex> lock(pipelines_mutex_);
        auto it = pipelines_.find(pipeline_id);
        if (it == pipelines_.end()) {
            res.status = 404;
            res.set_content(R"({"error":"Pipeline not found"})", "application/json");
            return;
        }

        it->second->stop();
        json response = {{"status", "ok"}, {"id", pipeline_id}, {"running", false}};
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' stopped via API", pipeline_id);
    } catch (const std::exception& e) {
        res.status = 500;
        json err = {{"error", e.what()}};
        res.set_content(err.dump(), "application/json");
    }
}

void ApiServer::handlePipelineDestroy(const httplib::Request& req, httplib::Response& res) {
    std::string pipeline_id = req.matches[1];
    
    std::lock_guard<std::mutex> lock(pipelines_mutex_);
    auto it = pipelines_.find(pipeline_id);
    if (it == pipelines_.end()) {
        res.status = 404;
        res.set_content(R"({"error":"Pipeline not found"})", "application/json");
        return;
    }

    it->second->stop();
    pipelines_.erase(it);
    
    json response = {{"status", "ok"}, {"id", pipeline_id}, {"message", "Pipeline destroyed"}};
    res.set_content(response.dump(), "application/json");
    LOG_INFO_FMT("Pipeline '{}' destroyed via API", pipeline_id);
}

void ApiServer::handlePipelineList(const httplib::Request& /*req*/, httplib::Response& res) {
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

void ApiServer::handlePipelineStatus(const httplib::Request& req, httplib::Response& res) {
    std::string pipeline_id = req.matches[1];
    
    std::lock_guard<std::mutex> lock(pipelines_mutex_);
    auto it = pipelines_.find(pipeline_id);
    if (it == pipelines_.end()) {
        res.status = 404;
        res.set_content(R"({"error":"Pipeline not found"})", "application/json");
        return;
    }

    json response = {
        {"id", pipeline_id},
        {"running", it->second->isRunning()}
    };
    res.set_content(response.dump(), "application/json");
}

void ApiServer::handleHealth(const httplib::Request& /*req*/, httplib::Response& res) {
    json health = {
        {"status", "healthy"},
        {"timestamp", std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count()}
    };
    res.set_content(health.dump(), "application/json");
}

} // namespace http
} // namespace ai_stream