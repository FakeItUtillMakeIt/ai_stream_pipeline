// src/http/request_log.cpp
#include "request_log.h"
#include "3rd_party/log_mgr/log_mgr.h"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <unordered_map>

namespace fs = std::filesystem;

namespace ai_stream {
namespace http {

namespace {
// 只保留安全字符做文件名，避免 id 里带 '/'、'..' 造成路径穿越。
std::string sanitize(const std::string& s)
{
    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (std::isalnum(static_cast<unsigned char>(c)) || c == '_' || c == '-' || c == '.')
            out += c;
        else
            out += '_';
    }
    if (out.empty()) out = "pipeline";
    return out;
}
}  // namespace

RequestLog::RequestLog(std::string state_dir) : state_dir_(std::move(state_dir))
{
    log_path_ = (fs::path(state_dir_) / "requests.log").string();
    std::error_code ec;
    fs::create_directories(fs::path(state_dir_) / "pipelines", ec);
    if (ec)
        LOG_WARN_FMT("[RequestLog] cannot create state dir {}: {}", state_dir_, ec.message());
    LOG_INFO_FMT("[RequestLog] state_dir={}", state_dir_);
}

std::string RequestLog::configPathImpl(const std::string& pipeline_id) const
{
    return (fs::path(state_dir_) / "pipelines" / (sanitize(pipeline_id) + ".json")).string();
}

std::string RequestLog::configPath(const std::string& pipeline_id) const
{
    return configPathImpl(pipeline_id);
}

std::string RequestLog::saveConfig(const std::string& pipeline_id, const nlohmann::json& build_body)
{
    const std::string path = configPathImpl(pipeline_id);
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(path, std::ios::trunc);
    if (!out) {
        LOG_WARN_FMT("[RequestLog] 无法写配置文件 {}", path);
        return {};
    }
    out << build_body.dump(2);
    out.flush();
    if (!out) {
        LOG_WARN_FMT("[RequestLog] 写配置文件失败 {}", path);
        return {};
    }
    return path;
}

void RequestLog::append(const std::string& action, const std::string& pipeline_id,
                        const std::string& config_path)
{
    nlohmann::json rec = {
        {"ts", std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
                   .count()},
        {"action", action},
        {"id", pipeline_id},
        {"config", config_path},
    };
    std::lock_guard<std::mutex> lock(mutex_);
    std::ofstream out(log_path_, std::ios::app);
    if (!out) {
        LOG_WARN_FMT("[RequestLog] 无法写历史 {}", log_path_);
        return;
    }
    out << rec.dump() << "\n";   // 一条一行，最新在最后一行
}

RequestLog::LastRecord RequestLog::last() const
{
    LastRecord r;
    std::lock_guard<std::mutex> lock(mutex_);
    std::ifstream in(log_path_);
    if (!in) return r;   // 无历史文件

    std::string line, last_line;
    while (std::getline(in, line)) {
        // 取最后一条非空行
        if (line.find_first_not_of(" \t\r\n") != std::string::npos)
            last_line = line;
    }
    if (last_line.empty()) return r;

    try {
        nlohmann::json j = nlohmann::json::parse(last_line);
        r.valid = true;
        r.action = j.value("action", "");
        r.id = j.value("id", "");
        r.config = j.value("config", "");
    } catch (const std::exception& e) {
        LOG_WARN_FMT("[RequestLog] 历史末行解析失败: {}", e.what());
    }
    return r;
}

std::vector<RequestLog::PipelineState> RequestLog::finalStates() const
{
    std::vector<RequestLog::PipelineState> out;
    std::lock_guard<std::mutex> lock(mutex_);
    std::ifstream in(log_path_);
    if (!in) return out;

    // 按 id 聚合：后出现的记录覆盖先前，得到每个 id 的最终状态；
    // 用 order 保留首次出现顺序，恢复时顺序稳定。
    std::unordered_map<std::string, size_t> index_of;
    std::string line;
    while (std::getline(in, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos)
            continue;
        try {
            nlohmann::json j = nlohmann::json::parse(line);
            std::string id = j.value("id", "");
            if (id.empty()) continue;
            PipelineState st;
            st.id = id;
            st.action = j.value("action", "");
            st.config = j.value("config", "");
            auto it = index_of.find(id);
            if (it == index_of.end()) {
                index_of[id] = out.size();
                out.push_back(std::move(st));
            } else {
                out[it->second] = std::move(st);   // 末态覆盖
            }
        } catch (...) {
            // 跳过坏行
        }
    }
    return out;
}

nlohmann::json RequestLog::loadConfig(const std::string& path) const
{
    std::lock_guard<std::mutex> lock(mutex_);
    std::ifstream in(path);
    if (!in) {
        LOG_WARN_FMT("[RequestLog] 配置文件不存在: {}", path);
        return nullptr;
    }
    try {
        nlohmann::json j;
        in >> j;
        return j;
    } catch (const std::exception& e) {
        LOG_WARN_FMT("[RequestLog] 配置文件解析失败 {}: {}", path, e.what());
        return nullptr;
    }
}

}  // namespace http
}  // namespace ai_stream
