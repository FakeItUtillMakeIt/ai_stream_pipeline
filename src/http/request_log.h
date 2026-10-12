// src/http/request_log.h
#pragma once

#include <mutex>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>

namespace ai_stream {
namespace http {

/**
 * @brief 平台请求持久化（掉线/重启后自动恢复用）
 *
 * - 历史文件 <state>/requests.log：文本，每行一条 JSON 记录
 *   {"ts":..,"action":"build|start|stop|delete","id":"..","config":"<state>/pipelines/<id>.json"}
 *   最新请求在最后一行。
 * - 配置文件 <state>/pipelines/<id>.json：保存 build 请求体（含 id + graph），
 *   恢复时据此重建管道。
 *
 * 线程安全：HTTP 多线程处理请求，内部加锁。
 */
class RequestLog {
public:
    explicit RequestLog(std::string state_dir);

    // 保存 build 请求体到 pipelines/<id>.json，返回文件路径（失败返回空串）。
    std::string saveConfig(const std::string& pipeline_id, const nlohmann::json& build_body);

    // 追加一条历史记录（action + id + 配置文件路径）。
    void append(const std::string& action, const std::string& pipeline_id,
                const std::string& config_path);

    struct LastRecord {
        bool valid = false;
        std::string action;
        std::string id;
        std::string config;
    };

    // 读取最后一条记录（无则 valid=false）。
    LastRecord last() const;

    // 每个 pipeline 的最终状态（按 id 聚合，取该 id 的最后一条记录）。
    struct PipelineState {
        std::string id;
        std::string action;    // build|start|stop|delete
        std::string config;    // 配置文件路径
    };
    // 扫描整个历史，返回每个 id 的最后状态（按首次出现顺序）。恢复用。
    std::vector<PipelineState> finalStates() const;

    // 读取某配置文件内容（恢复用）；失败返回 null json。
    nlohmann::json loadConfig(const std::string& path) const;

    const std::string& stateDir() const { return state_dir_; }

    // 该 pipeline 配置文件路径（<state>/pipelines/<id>.json）。
    std::string configPath(const std::string& pipeline_id) const;

private:
    std::string configPathImpl(const std::string& pipeline_id) const;

    std::string state_dir_;
    std::string log_path_;
    mutable std::mutex mutex_;
};

}  // namespace http
}  // namespace ai_stream
