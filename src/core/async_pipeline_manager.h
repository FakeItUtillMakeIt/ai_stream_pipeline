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
#include <unordered_set>
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
    AsyncPipelineManager();
    ~AsyncPipelineManager();

    /**
     * @brief 异步加载流水线配置
     * @param config_path JSON 配置文件路径
     * @return 流水线 ID，空字符串表示失败
     */
    std::string loadPipelineAsync(const std::string& config_path);

    /**
     * @brief 异步启动流水线
     * @param pipeline_id 流水线 ID
     * @return 是否成功提交启动请求
     */
    bool startPipelineAsync(const std::string& pipeline_id);

    /**
     * @brief 异步停止流水线
     * @param pipeline_id 流水线 ID
     * @return 是否成功提交停止请求
     */
    bool stopPipelineAsync(const std::string& pipeline_id);

    /**
     * @brief 获取流水线状态
     * @param pipeline_id 流水线 ID
     * @return 流水线状态 (true=running, false=stopped)
     */
    bool getPipelineState(const std::string& pipeline_id) const;

    /**
     * @brief 获取所有流水线 ID
     * @return 流水线 ID 列表
     */
    std::vector<std::string> getAllPipelineIds() const;

    /**
     * @brief 移除流水线
     * @param pipeline_id 流水线 ID
     * @return 是否成功移除
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
        std::shared_ptr<std::promise<std::string>> promise; // 用于 LOAD 任务返回 pipeline_id
    };

    void taskWorker();
    void processTask(PipelineTask& task);
    void monitorResourceUsage();

    mutable std::mutex mutex_;
    mutable std::mutex task_mutex_;
    std::condition_variable task_cv_;
    std::queue<std::unique_ptr<PipelineTask>> task_queue_;
    std::unordered_map<std::string, std::shared_ptr<Pipeline>> pipelines_;
    std::unordered_set<std::string> pending_operations_;

    std::thread worker_thread_;
    std::thread monitor_thread_;
    std::atomic<bool> running_{true};

    mutable std::atomic<int> gpu_memory_mb_{0};
    mutable std::atomic<int> cpu_usage_percent_{0};
};

} // namespace core
} // namespace ai_stream
