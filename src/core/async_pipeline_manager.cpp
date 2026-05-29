// src/core/async_pipeline_manager.cpp
// 【加速优化】异步流水线管理器 - 支持动态批处理和资源调度
#include "async_pipeline_manager.h"
#include "ai_stream/core/metrics.h"
#include "3rd_party/log_mgr/log_mgr.h"
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <algorithm>

#ifdef WITH_CUDA
#include <cuda_runtime.h>
#endif

namespace ai_stream {
namespace core {

AsyncPipelineManager::AsyncPipelineManager() {
    worker_thread_ = std::thread(&AsyncPipelineManager::taskWorker, this);
    monitor_thread_ = std::thread(&AsyncPipelineManager::monitorResourceUsage, this);
    LOG_INFO_FMT("[AsyncPipelineManager] Initialized");
}

AsyncPipelineManager::~AsyncPipelineManager() {
    running_ = false;
    task_cv_.notify_all();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }
    if (monitor_thread_.joinable()) {
        monitor_thread_.join();
    }

    // 停止所有流水线
    for (auto& [id, pipeline] : pipelines_) {
        if (pipeline && pipeline->isRunning()) {
            pipeline->stop();
        }
    }

    LOG_INFO_FMT("[AsyncPipelineManager] Destroyed");
}

std::string AsyncPipelineManager::loadPipelineAsync(const std::string& config_path) {
    auto task = std::make_unique<PipelineTask>();
    task->type = PipelineTask::Type::LOAD;
    task->config_path = config_path;
    task->promise = std::make_shared<std::promise<std::string>>();

    auto future = task->promise->get_future();
    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_queue_.push(std::move(task));
    }
    task_cv_.notify_one();

    try {
        return future.get();
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[AsyncPipelineManager] Failed to load pipeline: {}", e.what());
        return "";
    }
}

bool AsyncPipelineManager::startPipelineAsync(const std::string& pipeline_id) {
    auto task = std::make_unique<PipelineTask>();
    task->type = PipelineTask::Type::START;
    task->pipeline_id = pipeline_id;

    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_queue_.push(std::move(task));
    }
    task_cv_.notify_one();
    return true;
}

bool AsyncPipelineManager::stopPipelineAsync(const std::string& pipeline_id) {
    auto task = std::make_unique<PipelineTask>();
    task->type = PipelineTask::Type::STOP;
    task->pipeline_id = pipeline_id;

    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_queue_.push(std::move(task));
    }
    task_cv_.notify_one();
    return true;
}

bool AsyncPipelineManager::removePipeline(const std::string& pipeline_id) {
    auto task = std::make_unique<PipelineTask>();
    task->type = PipelineTask::Type::REMOVE;
    task->pipeline_id = pipeline_id;

    {
        std::lock_guard<std::mutex> lock(task_mutex_);
        task_queue_.push(std::move(task));
    }
    task_cv_.notify_one();
    return true;
}

bool AsyncPipelineManager::getPipelineState(const std::string& pipeline_id) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = pipelines_.find(pipeline_id);
    if (it != pipelines_.end() && it->second) {
        return it->second->isRunning();
    }
    return false;
}

std::vector<std::string> AsyncPipelineManager::getAllPipelineIds() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::vector<std::string> ids;
    ids.reserve(pipelines_.size());
    for (const auto& [id, _] : pipelines_) {
        ids.push_back(id);
    }
    return ids;
}

AsyncPipelineManager::ResourceStats AsyncPipelineManager::getResourceStats() const {
    ResourceStats stats;
    stats.gpu_memory_mb = gpu_memory_mb_.load();
    stats.cpu_usage_percent = cpu_usage_percent_.load();

    std::lock_guard<std::mutex> lock(mutex_);
    stats.total_pipelines = static_cast<int>(pipelines_.size());
    for (const auto& [_, pipeline] : pipelines_) {
        if (pipeline && pipeline->isRunning()) {
            stats.active_pipelines++;
        }
    }
    return stats;
}

void AsyncPipelineManager::taskWorker() {
    while (running_) {
        std::unique_ptr<PipelineTask> task;
        {
            std::unique_lock<std::mutex> lock(task_mutex_);
            task_cv_.wait(lock, [this] { return !task_queue_.empty() || !running_; });

            if (!running_ && task_queue_.empty()) {
                break;
            }

            task = std::move(task_queue_.front());
            task_queue_.pop();
        }

        processTask(*task);
    }
}

void AsyncPipelineManager::processTask(PipelineTask& task) {
    try {
        switch (task.type) {
            case PipelineTask::Type::LOAD: {
                LOG_INFO_FMT("[AsyncPipelineManager] Loading pipeline from: {}", task.config_path);

                // 读取配置文件
                std::ifstream file(task.config_path);
                if (!file.is_open()) {
                    LOG_ERROR_FMT("[AsyncPipelineManager] Failed to open config: {}", task.config_path);
                    if (task.promise) task.promise->set_value("");
                    return;
                }

                std::stringstream buffer;
                buffer << file.rdbuf();
                nlohmann::json config;
                try {
                    config = nlohmann::json::parse(buffer.str());
                } catch (const std::exception& e) {
                    LOG_ERROR_FMT("[AsyncPipelineManager] Failed to parse config: {}", e.what());
                    if (task.promise) task.promise->set_value("");
                    return;
                }

                // 创建流水线
                std::string pipeline_id = config.value("id", "pipeline_" + std::to_string(std::hash<std::string>{}(task.config_path)));
                auto pipeline = std::make_shared<Pipeline>(pipeline_id);

                // 【加速优化】动态批处理配置
                if (config.contains("optimization")) {
                    auto& opt_cfg = config["optimization"];

                    // 根据资源自动调整批处理大小
                    if (opt_cfg.contains("auto_batch") && opt_cfg["auto_batch"].get<bool>()) {
                        int recommended_batch = 1;

                        // 简单启发式：根据 GPU 内存推荐批处理大小
                        size_t free_mem_mb = 0;
                        size_t total_mem_mb = 0;

#ifdef WITH_CUDA
                        int device;
                        cudaGetDevice(&device);
                        size_t free_mem, total_mem;
                        cudaMemGetInfo(&free_mem, &total_mem);
                        free_mem_mb = free_mem / (1024 * 1024);
                        total_mem_mb = total_mem / (1024 * 1024);
#endif

                        if (free_mem_mb > 2000) {
                            recommended_batch = 8;
                        } else if (free_mem_mb > 1000) {
                            recommended_batch = 4;
                        } else if (free_mem_mb > 500) {
                            recommended_batch = 2;
                        }

                        LOG_INFO_FMT("[AsyncPipelineManager] Auto-batch: recommended={}, free_mem={}MB",
                                     recommended_batch, free_mem_mb);

                        // 应用到推理节点
                        if (config.contains("nodes") && config["nodes"].is_array()) {
                            for (auto& node_cfg : config["nodes"]) {
                                if (node_cfg.contains("type") &&
                                    node_cfg["type"].get<std::string>().find("infer") != std::string::npos) {

                                    if (!node_cfg.contains("params")) {
                                        node_cfg["params"] = nlohmann::json::object();
                                    }

                                    if (!node_cfg["params"].contains("detector_config")) {
                                        node_cfg["params"]["detector_config"] = nlohmann::json::object();
                                    }

                                    node_cfg["params"]["detector_config"]["batch_size"] = recommended_batch;
                                    LOG_INFO_FMT("[AsyncPipelineManager] Set batch_size={} for node: {}",
                                                 recommended_batch, node_cfg.value("id", "unknown"));
                                }
                            }
                        }
                    }
                }

                if (!pipeline->buildFromJson(config)) {
                    LOG_ERROR_FMT("[AsyncPipelineManager] Failed to build pipeline: {}", pipeline_id);
                    if (task.promise) task.promise->set_value("");
                    return;
                }

                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    pipelines_[pipeline_id] = pipeline;
                }

                LOG_INFO_FMT("[AsyncPipelineManager] Pipeline loaded: {}", pipeline_id);
                if (task.promise) task.promise->set_value(pipeline_id);
                break;
            }

            case PipelineTask::Type::START: {
                LOG_INFO_FMT("[AsyncPipelineManager] Starting pipeline: {}", task.pipeline_id);

                std::shared_ptr<Pipeline> pipeline;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto it = pipelines_.find(task.pipeline_id);
                    if (it != pipelines_.end()) {
                        pipeline = it->second;
                    }
                }

                if (pipeline) {
                    if (pipeline->start()) {
                        LOG_INFO_FMT("[AsyncPipelineManager] Pipeline started: {}", task.pipeline_id);
                    } else {
                        LOG_ERROR_FMT("[AsyncPipelineManager] Failed to start pipeline: {}", task.pipeline_id);
                    }
                } else {
                    LOG_ERROR_FMT("[AsyncPipelineManager] Pipeline not found: {}", task.pipeline_id);
                }
                break;
            }

            case PipelineTask::Type::STOP: {
                LOG_INFO_FMT("[AsyncPipelineManager] Stopping pipeline: {}", task.pipeline_id);

                std::shared_ptr<Pipeline> pipeline;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto it = pipelines_.find(task.pipeline_id);
                    if (it != pipelines_.end()) {
                        pipeline = it->second;
                    }
                }

                if (pipeline) {
                    pipeline->stop();
                    LOG_INFO_FMT("[AsyncPipelineManager] Pipeline stopped: {}", task.pipeline_id);
                } else {
                    LOG_ERROR_FMT("[AsyncPipelineManager] Pipeline not found: {}", task.pipeline_id);
                }
                break;
            }

            case PipelineTask::Type::REMOVE: {
                LOG_INFO_FMT("[AsyncPipelineManager] Removing pipeline: {}", task.pipeline_id);

                std::shared_ptr<Pipeline> pipeline;
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    auto it = pipelines_.find(task.pipeline_id);
                    if (it != pipelines_.end()) {
                        pipeline = it->second;
                        pipelines_.erase(it);
                    }
                }

                if (pipeline) {
                    if (pipeline->isRunning()) {
                        pipeline->stop();
                    }
                    LOG_INFO_FMT("[AsyncPipelineManager] Pipeline removed: {}", task.pipeline_id);
                } else {
                    LOG_ERROR_FMT("[AsyncPipelineManager] Pipeline not found: {}", task.pipeline_id);
                }
                break;
            }
        }
    } catch (const std::exception& e) {
        LOG_ERROR_FMT("[AsyncPipelineManager] Exception in task processing: {}", e.what());
    }
}

void AsyncPipelineManager::monitorResourceUsage() {
    while (running_) {
        // 监控 GPU 内存使用
#ifdef WITH_CUDA
        size_t free_mem, total_mem;
        cudaError_t err = cudaMemGetInfo(&free_mem, &total_mem);
        if (err == cudaSuccess) {
            size_t used_mem_mb = (total_mem - free_mem) / (1024 * 1024);
            gpu_memory_mb_.store(static_cast<int>(used_mem_mb));
        }
#endif

        // 简单的 CPU 使用率估算
        static auto last_time = std::chrono::steady_clock::now();
        static auto last_idle = 0ULL;
        static auto last_total = 0ULL;

        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();

        if (elapsed > 1000) { // 每秒更新一次
#ifdef __linux__
            std::ifstream stat_file("/proc/stat");
            if (stat_file.is_open()) {
                std::string line;
                if (std::getline(stat_file, line)) {
                    std::istringstream iss(line);
                    std::string cpu_label;
                    unsigned long long user, nice, system, idle, iowait, irq, softirq, steal;
                    iss >> cpu_label >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;

                    unsigned long long current_idle = idle + iowait;
                    unsigned long long current_total = user + nice + system + idle + iowait + irq + softirq + steal;

                    if (last_total != 0) {
                        unsigned long long total_diff = current_total - last_total;
                        unsigned long long idle_diff = current_idle - last_idle;
                        unsigned long long cpu_usage = 100 * (total_diff - idle_diff) / total_diff;
                        cpu_usage_percent_.store(static_cast<int>(cpu_usage));
                    }

                    last_idle = current_idle;
                    last_total = current_total;
                }
            }
#endif
            last_time = now;
        }

        // Feed resource stats into MetricsCollector
        auto& mc = MetricsCollector::instance();
        mc.setGpuMemoryMb(gpu_memory_mb_.load());
        mc.setCpuUsagePercent(cpu_usage_percent_.load());
        {
            std::lock_guard<std::mutex> lock(mutex_);
            int active = 0;
            for (const auto& [_, p] : pipelines_) {
                if (p && p->isRunning()) active++;
            }
            mc.setActivePipelines(active);
            mc.setTotalPipelines(static_cast<int>(pipelines_.size()));
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

} // namespace core
} // namespace ai_stream
