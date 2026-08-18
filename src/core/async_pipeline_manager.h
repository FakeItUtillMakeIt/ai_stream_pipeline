// src/core/async_pipeline_manager.h
// 【加速优化】异步流水线管理器 - 支持动态批处理和资源调度
#pragma once

#include "ai_stream/core/pipeline.h"
#include <nlohmann/json.hpp>
#include <thread>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <unordered_map>
#include <future>
#include <string>

namespace ai_stream {
namespace core {

/**
 * @brief 异步流水线管理器
 *
 * 提供高级流水线调度功能：
 * - 动态批处理优化
 * - 资源感知调度
 * - 异步启动/停止
 * - 边缘设备资源管理
 */
class AsyncPipelineManager {
public:
    /**
     * @brief 流水线生命周期状态
     */
    enum class PipelineState {
        UNKNOWN,      // 不存在
        LOADING,      // 构建任务已提交，正在异步构建
        LOAD_FAILED,  // 构建失败
        STOPPED,      // 已构建未运行（或已停止）
        RUNNING       // 运行中
    };
    static std::string stateToString(PipelineState state);

    AsyncPipelineManager();
    ~AsyncPipelineManager();

    /**
     * @brief 异步加载流水线配置（文件路径版，阻塞直到加载完成）
     * @param config_path JSON 配置文件路径
     * @return 流水线 ID，空字符串表示失败
     */
    std::string loadPipelineAsync(const std::string& config_path);

    /**
     * @brief 异步加载流水线配置（JSON 内联版，立即返回）
     *
     * 配置格式与 HTTP API 一致：{"id": "...", "graph": {"nodes": [...], "edges": [...]}}，
     * 也兼容无 "graph" 包裹的扁平格式。构建在任务线程中异步执行，
     * 通过 getPipelineState() 轮询状态（LOADING -> STOPPED / LOAD_FAILED）。
     *
     * @param config 完整配置 JSON
     * @return 流水线 ID，空字符串表示配置无效或 ID 重复
     */
    std::string loadPipelineFromJsonAsync(const nlohmann::json& config);

    /**
     * @brief 异步启动流水线
     * @param pipeline_id 流水线 ID
     * @return 是否成功提交启动请求（false 表示流水线不存在）
     */
    bool startPipelineAsync(const std::string& pipeline_id);

    /**
     * @brief 异步停止流水线
     * @param pipeline_id 流水线 ID
     * @return 是否成功提交停止请求（false 表示流水线不存在）
     */
    bool stopPipelineAsync(const std::string& pipeline_id);

    /**
     * @brief 获取流水线运行状态
     * @param pipeline_id 流水线 ID
     * @return 流水线状态 (true=running, false=stopped)
     */
    bool getPipelineState(const std::string& pipeline_id) const;

    /**
     * @brief 获取流水线生命周期状态
     * @param pipeline_id 流水线 ID
     * @return 生命周期状态枚举
     */
    PipelineState getPipelineLifecycleState(const std::string& pipeline_id) const;

    /**
     * @brief 流水线是否存在（含正在异步构建中的）
     */
    bool hasPipeline(const std::string& pipeline_id) const;

    /**
     * @brief 获取所有流水线 ID
     * @return 流水线 ID 列表
     */
    std::vector<std::string> getAllPipelineIds() const;

    /**
     * @brief 移除流水线
     * @param pipeline_id 流水线 ID
     * @return 是否成功提交移除请求（false 表示流水线不存在）
     */
    bool removePipeline(const std::string& pipeline_id);

    /**
     * @brief 获取系统资源使用情况
     * @return 资源使用统计
     */
    struct ResourceStats {
        int gpu_memory_mb = 0;
        int cpu_usage_percent = 0;
        int active_pipelines = 0;
        int total_pipelines = 0;
    };
    ResourceStats getResourceStats() const;

private:
    struct PipelineTask {
        enum class Type { LOAD, START, STOP, REMOVE } type;
        std::string pipeline_id;
        std::string config_path;
        nlohmann::json config;                       // 内联配置（非空时优先于 config_path）
        std::shared_ptr<std::promise<std::string>> promise; // 用于阻塞式 LOAD 任务返回 pipeline_id
    };

    void taskWorker();
    void processTask(PipelineTask& task);
    void processLoadTask(PipelineTask& task);
    void setLifecycleState(const std::string& pipeline_id, PipelineState state);
    void monitorResourceUsage();

    mutable std::mutex mutex_;
    mutable std::mutex task_mutex_;
    std::condition_variable task_cv_;
    std::queue<std::unique_ptr<PipelineTask>> task_queue_;
    std::unordered_map<std::string, std::shared_ptr<Pipeline>> pipelines_;
    std::unordered_map<std::string, PipelineState> states_;

    std::thread worker_thread_;
    std::thread monitor_thread_;
    std::atomic<bool> running_{true};

    mutable std::atomic<int> gpu_memory_mb_{0};
    mutable std::atomic<int> cpu_usage_percent_{0};
};

} // namespace core
} // namespace ai_stream
