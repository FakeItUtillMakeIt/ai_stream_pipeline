// src/http/api_server.cpp
#include "api_server.h"
#include "ai_stream/core/metrics.h"
#include "ai_stream/hal/backend_diagnostics.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <nlohmann/json.hpp>
#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <cctype>
#include <fstream>
#include <sstream>

using json = nlohmann::json;
namespace fs = std::filesystem;

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
    // 请求持久化状态目录（默认 ./api_state，可用 AI_STREAM_STATE_DIR 覆盖）
    {
        const char* sd = std::getenv("AI_STREAM_STATE_DIR");
        reqlog_ = std::make_unique<RequestLog>(sd && *sd ? sd : "./api_state");
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

void ApiServer::recordRequest(const std::string& action, const std::string& pipeline_id,
                              const json* build_body) {
    if (!reqlog_ || pipeline_id.empty())
        return;
    std::string cfg;
    if (action == "build" && build_body) {
        cfg = reqlog_->saveConfig(pipeline_id, *build_body);   // 存配置文件
    } else {
        cfg = reqlog_->configPath(pipeline_id);                // 复用 build 时的配置
    }
    reqlog_->append(action, pipeline_id, cfg);
}

void ApiServer::recoverFromState() {
    if (!reqlog_) return;
    auto states = reqlog_->finalStates();
    if (states.empty()) {
        LOG_INFO("[Recover] 无历史请求记录，等待平台下发");
        return;
    }

    int rebuilt = 0, started = 0, skipped = 0;
    for (const auto& st : states) {
        // 已销毁的管道不复活
        if (st.action == "delete") {
            LOG_INFO_FMT("[Recover] '{}' 末态 delete，跳过", st.id);
            skipped++;
            continue;
        }
        json cfg = reqlog_->loadConfig(st.config);
        if (cfg.is_null() || !cfg.is_object() || !cfg.contains("graph")) {
            LOG_WARN_FMT("[Recover] '{}' 配置文件无效，跳过：{}", st.id, st.config);
            skipped++;
            continue;
        }
        // build/stop → 只重建不启动；start → 重建并启动
        bool should_start = (st.action == "start");

        try {
            if (async_mode_ && manager_) {
                if (!manager_->hasPipeline(st.id))
                    manager_->loadPipelineFromJsonAsync(cfg);
                if (should_start)
                    manager_->startPipelineAsync(st.id);
            } else {
                std::shared_ptr<core::Pipeline> pipeline;
                {
                    std::lock_guard<std::mutex> lock(pipelines_mutex_);
                    auto it = pipelines_.find(st.id);
                    if (it != pipelines_.end()) pipeline = it->second;
                }
                if (!pipeline) {
                    pipeline = std::make_shared<core::Pipeline>(st.id);
                    if (!pipeline->buildFromJson(cfg["graph"])) {
                        LOG_WARN_FMT("[Recover] '{}' 重建失败", st.id);
                        skipped++;
                        continue;
                    }
                    std::lock_guard<std::mutex> lock(pipelines_mutex_);
                    pipelines_[st.id] = pipeline;
                }
                if (should_start && !pipeline->start()) {
                    LOG_WARN_FMT("[Recover] '{}' 启动失败", st.id);
                    should_start = false;   // 启动没成功，不计入 started
                }
            }
            rebuilt++;
            extractMeta(st.id, cfg);   // 恢复的管道同样登记 play_url / evidence 目录
            if (should_start) started++;
            LOG_INFO_FMT("[Recover] '{}' 末态={} → 重建{}", st.id, st.action,
                         should_start ? "+启动" : "（不启动）");
        } catch (const std::exception& e) {
            LOG_ERROR_FMT("[Recover] '{}' 恢复异常: {}", st.id, e.what());
            skipped++;
        }
    }
    LOG_INFO_FMT("[Recover] 完成：重建 {}（含启动 {}），跳过 {}", rebuilt, started, skipped);
}

void ApiServer::handleSaveConfig(const httplib::Request& req, httplib::Response& res) {
    json body;
    if (!parseJsonBody(req, res, body)) return;
    try {
        std::string pipeline_id = idOf(body);
        if (pipeline_id.empty() || !body.contains("graph")) {
            res.status = 400;
            res.set_content(R"({"error":"Missing pipeline id or graph"})", "application/json");
            return;
        }
        if (!reqlog_) {
            res.status = 500;
            res.set_content(R"({"error":"RequestLog unavailable"})", "application/json");
            return;
        }
        std::string path = reqlog_->saveConfig(pipeline_id, body);
        if (path.empty()) {
            res.status = 500;
            res.set_content(R"({"error":"Failed to write config"})", "application/json");
            return;
        }
        json response = {{"status", "ok"}, {"id", pipeline_id}, {"config", path}};
        res.set_content(response.dump(), "application/json");
        LOG_INFO_FMT("Pipeline '{}' config saved to {}", pipeline_id, path);
    } catch (const std::exception& e) {
        res.status = 500;
        res.set_content(json{{"error", e.what()}}.dump(), "application/json");
    }
}

void ApiServer::extractMeta(const std::string& pipeline_id, const json& body) {
    PipelineMeta m;
    try {
        if (body.contains("graph") && body["graph"].contains("nodes") && body["graph"]["nodes"].is_array()) {
            for (const auto& n : body["graph"]["nodes"]) {
                if (!n.is_object()) continue;
                const std::string type = n.value("type", "");
                json params = n.contains("params") && n["params"].is_object() ? n["params"] : json::object();
                // 播放地址：取自输出侧 sink 节点的 play_url
                if ((type == "rtmp_sink" || type == "mp4_save" || type == "hls_sink") &&
                    params.contains("play_url") && params["play_url"].is_string())
                    m.play_url = params["play_url"].get<std::string>();
                // 证据目录：evidence 节点 video/snapshot 的 output_dir
                if (type == "evidence") {
                    auto addDir = [&](const json& sub) {
                        if (sub.is_object() && sub.contains("output_dir") && sub["output_dir"].is_string()) {
                            std::string d = sub["output_dir"].get<std::string>();
                            if (!d.empty()) m.evidence_dirs.push_back(d);
                        }
                    };
                    if (params.contains("video")) addDir(params["video"]);
                    if (params.contains("snapshot")) addDir(params["snapshot"]);
                }
            }
        }
    } catch (const std::exception& e) {
        LOG_WARN_FMT("extractMeta '{}' 异常: {}", pipeline_id, e.what());
    }
    std::lock_guard<std::mutex> lk(meta_mutex_);
    meta_[pipeline_id] = std::move(m);
}

static std::string evidenceKind(const std::string& name) {
    auto dot = name.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : name.substr(dot + 1);
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    if (ext == "jpg" || ext == "jpeg" || ext == "png") return "image";
    if (ext == "mp4" || ext == "mkv" || ext == "webm" || ext == "ts") return "video";
    return "other";
}
static const char* evidenceMime(const std::string& name) {
    auto dot = name.find_last_of('.');
    std::string ext = dot == std::string::npos ? "" : name.substr(dot + 1);
    for (auto& c : ext) c = static_cast<char>(std::tolower(c));
    if (ext == "jpg" || ext == "jpeg") return "image/jpeg";
    if (ext == "png") return "image/png";
    if (ext == "mp4") return "video/mp4";
    if (ext == "webm") return "video/webm";
    return "application/octet-stream";
}

void ApiServer::handleEvidenceList(const httplib::Request& req, httplib::Response& res) {
    const std::string id = req.get_param_value("pipeline");
    PipelineMeta m;
    {
        std::lock_guard<std::mutex> lk(meta_mutex_);
        auto it = meta_.find(id);
        if (it != meta_.end()) m = it->second;
    }
    json files = json::array();
    std::error_code ec;
    for (size_t i = 0; i < m.evidence_dirs.size(); ++i) {
        fs::path dir = fs::absolute(m.evidence_dirs[i], ec);  // 相对程序运行目录解析
        if (!fs::is_directory(dir, ec)) continue;
        for (const auto& entry : fs::directory_iterator(dir, ec)) {
            if (!entry.is_regular_file()) continue;
            const std::string name = entry.path().filename().string();
            json f;
            f["dir_index"] = static_cast<int>(i);
            f["name"] = name;
            f["kind"] = evidenceKind(name);
            f["size"] = static_cast<long long>(entry.file_size(ec));
            files.push_back(std::move(f));
        }
    }
    json resp = {{"pipeline", id}, {"play_url", m.play_url}, {"files", files}};
    res.set_content(resp.dump(), "application/json");
}

void ApiServer::handleEvidenceFile(const httplib::Request& req, httplib::Response& res) {
    const std::string id = req.get_param_value("pipeline");
    const std::string name = req.get_param_value("name");
    int dir_index = std::atoi(req.get_param_value("dir").c_str());
    // 安全：文件名不得含路径分隔/..
    if (name.empty() || name.find('/') != std::string::npos || name.find('\\') != std::string::npos ||
        name.find("..") != std::string::npos) {
        res.status = 400; res.set_content(R"({"error":"bad name"})", "application/json"); return;
    }
    PipelineMeta m;
    {
        std::lock_guard<std::mutex> lk(meta_mutex_);
        auto it = meta_.find(id);
        if (it != meta_.end()) m = it->second;
    }
    if (dir_index < 0 || dir_index >= static_cast<int>(m.evidence_dirs.size())) {
        res.status = 400; res.set_content(R"({"error":"bad dir"})", "application/json"); return;
    }
    std::error_code ec;
    fs::path base = fs::absolute(m.evidence_dirs[dir_index], ec);
    fs::path full = base / name;
    std::string absFull = fs::absolute(full, ec).lexically_normal().string();
    std::string absBase = base.lexically_normal().string();
    if (absFull.compare(0, absBase.size(), absBase) != 0) {  // 越界防护
        res.status = 403; res.set_content(R"({"error":"forbidden"})", "application/json"); return;
    }
    std::ifstream in(absFull, std::ios::binary | std::ios::ate);
    if (!in) { res.status = 404; res.set_content(R"({"error":"not found"})", "application/json"); return; }
    const long long fsize = static_cast<long long>(in.tellg());
    res.set_header("Accept-Ranges", "bytes");

    // 解析 Range: bytes=start-end（支持 mp4 播放/拖动；浏览器先发 bytes=0-）
    long long start = 0, end = fsize - 1;
    bool partial = false;
    if (req.has_header("Range")) {
        std::string r = req.get_header_value("Range");
        const std::string pfx = "bytes=";
        if (r.compare(0, pfx.size(), pfx) == 0) {
            std::string spec = r.substr(pfx.size());
            auto comma = spec.find(',');
            if (comma != std::string::npos) spec = spec.substr(0, comma);  // 只取第一段
            auto dash = spec.find('-');
            std::string s1 = dash == std::string::npos ? spec : spec.substr(0, dash);
            std::string s2 = dash == std::string::npos ? "" : spec.substr(dash + 1);
            if (s1.empty() && !s2.empty()) {                 // bytes=-N（末尾 N 字节）
                long long n = std::atoll(s2.c_str());
                start = std::max<long long>(0, fsize - n); end = fsize - 1;
            } else {
                if (!s1.empty()) start = std::atoll(s1.c_str());
                if (!s2.empty()) end = std::atoll(s2.c_str());
            }
            if (start > end || start >= fsize) {             // 越界
                res.status = 416;
                res.set_header("Content-Range", "bytes */" + std::to_string(fsize));
                res.set_content("", "text/plain"); return;
            }
            end = std::min<long long>(end, fsize - 1);
            partial = true;
        }
    }
    long long len = end - start + 1;
    std::vector<char> buf(static_cast<size_t>(len));
    in.seekg(start, std::ios::beg);
    in.read(buf.data(), len);
    buf.resize(static_cast<size_t>(in.gcount()));
    if (partial) {
        res.status = 206;
        res.set_header("Content-Range",
                       "bytes " + std::to_string(start) + "-" + std::to_string(end) + "/" + std::to_string(fsize));
    }
    res.set_content(std::string(buf.data(), buf.size()), evidenceMime(name));
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
    // 保存管道配置文件（编辑器"保存到服务器"）：写 <state>/pipelines/<id>.json
    server_.Post("/api/v1/pipeline/save-config", [this](const auto& req, auto& res) {
        handleSaveConfig(req, res);
    });
    // 证据查看（预览页）
    server_.Get("/api/v1/evidence/list", [this](const auto& req, auto& res) {
        handleEvidenceList(req, res);
    });
    server_.Get("/api/v1/evidence/file", [this](const auto& req, auto& res) {
        handleEvidenceFile(req, res);
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

    // ===== 静态托管：Web 管道编辑器（同源，免 CORS）=====
    // 根目录用 AI_STREAM_WEB_ROOT 指定，默认 ./web/pipeline-editor/dist
    {
        const char* wr = std::getenv("AI_STREAM_WEB_ROOT");
        std::string root = (wr && *wr) ? wr : "./web/pipeline-editor/dist";
        std::error_code ec;
        if (fs::exists(root, ec) && fs::is_directory(root, ec)) {
            // 静态资源（js/css/图标等）由 mount 提供
            server_.set_mount_point("/", root);
            // SPA 根路径回退到 index.html
            server_.Get("/", [root](const httplib::Request&, httplib::Response& res) {
                std::ifstream idx((fs::path(root) / "index.html").string(), std::ios::binary);
                if (idx) { std::stringstream ss; ss << idx.rdbuf(); res.set_content(ss.str(), "text/html"); }
                else { res.status = 404; res.set_content("index.html not found", "text/plain"); }
            });
            LOG_INFO_FMT("Serving web editor from {}", root);
        } else {
            LOG_WARN_FMT("web root 不存在，跳过静态托管: {}（先构建编辑器再放到该路径）", root);
        }
    }
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
    // 服务就绪后，尝试按历史最后一条请求自动恢复（仅 start 才恢复）
    if (server_.is_running())
        recoverFromState();
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

        // replace=true：若已存在则先停止并销毁旧管道，再重建（编辑器"覆盖重建"）
        const bool replace = body.value("replace", false);
        {
            std::shared_ptr<core::Pipeline> old;
            {
                std::lock_guard<std::mutex> lock(pipelines_mutex_);
                auto it = pipelines_.find(pipeline_id);
                if (it != pipelines_.end()) {
                    if (!replace) {
                        res.status = 409;
                        res.set_content(R"({"error":"Pipeline id already exists"})", "application/json");
                        return;
                    }
                    old = it->second;
                    pipelines_.erase(it);
                }
            }
            if (old) {
                old->stop();
                core::MetricsCollector::instance().reset(pipeline_id);
                recordRequest("delete", pipeline_id);
                LOG_INFO_FMT("Pipeline '{}' replaced: 旧管道已停止并销毁", pipeline_id);
            }
        }
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
        recordRequest("build", pipeline_id, &body);
        extractMeta(pipeline_id, body);
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
        recordRequest("start", pipeline_id);
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
        recordRequest("stop", pipeline_id);
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
        recordRequest("delete", pipeline_id);
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
            json item = {{"id", id}, {"running", pipeline->isRunning()}};
            {
                std::lock_guard<std::mutex> mlk(meta_mutex_);
                auto mit = meta_.find(id);
                if (mit != meta_.end()) {
                    item["play_url"] = mit->second.play_url;
                    item["evidence_dirs"] = mit->second.evidence_dirs;
                }
            }
            list.push_back(item);
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

        // replace=true：已存在则先销毁旧管道再重建
        if (body.value("replace", false) && manager_->hasPipeline(idOf(body))) {
            if (manager_->removePipeline(idOf(body))) {
                recordRequest("delete", idOf(body));
                LOG_INFO_FMT("Pipeline '{}' replaced: 旧管道已提交销毁", idOf(body));
            }
        }

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
        recordRequest("build", pipeline_id, &body);
        extractMeta(pipeline_id, body);
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
        recordRequest("start", pipeline_id);
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
        recordRequest("stop", pipeline_id);
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
        recordRequest("delete", pipeline_id);
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
            json item = {
                {"id", id},
                {"state", core::AsyncPipelineManager::stateToString(state)},
                {"running", manager_->getPipelineState(id)}
            };
            {
                std::lock_guard<std::mutex> mlk(meta_mutex_);
                auto mit = meta_.find(id);
                if (mit != meta_.end()) {
                    item["play_url"] = mit->second.play_url;
                    item["evidence_dirs"] = mit->second.evidence_dirs;
                }
            }
            list.push_back(item);
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
            double avg = m.total_packets > 0
                ? static_cast<double>(m.total_latency_ms) / m.total_packets : 0.0;
            // min 用哨兵（UINT64_MAX）表示未设置，对外统一输出 0
            uint64_t minv = m.total_packets > 0 ? m.min_latency_ms : 0;
            nodes_json.push_back({
                {"node_name", m.node_name},
                {"total_packets", m.total_packets},
                {"dropped_packets", m.dropped_packets},
                {"latency_ms", {
                    {"avg", avg},
                    {"min", minv},
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
