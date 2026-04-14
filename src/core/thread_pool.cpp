// src/core/thread_pool.cpp
#include "ai_stream/core/thread_pool.h"
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace core {

ThreadPool::ThreadPool(size_t num_threads) : stop_(false) {
    for (size_t i = 0; i < num_threads; ++i) {
        workers_.emplace_back([this] {
            while (true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);
                    condition_.wait(lock, [this] { return stop_ || !tasks_.empty(); });
                    if (stop_ && tasks_.empty()) {
                        return;
                    }
                    task = std::move(tasks_.front());
                    tasks_.pop();
                }
                task();
            }
        });
    }
    LOG_DEBUG_FMT("ThreadPool created with {} threads", num_threads);
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        stop_ = true;
    }
    condition_.notify_all();
    for (std::thread& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
    LOG_DEBUG_FMT("ThreadPool destroyed");
}

void ThreadPool::enqueue(std::function<void()> task) {
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (stop_) {
            LOG_WARN_FMT("Enqueue on stopped ThreadPool");
            return;
        }
        tasks_.emplace(std::move(task));
    }
    condition_.notify_one();
}

} // namespace core
} // namespace ai_stream