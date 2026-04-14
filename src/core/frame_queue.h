// src/core/frame_queue.h
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <atomic>

namespace ai_stream {
namespace core {

/**
 * @brief 线程安全的有界阻塞队列
 * 
 * 用于节点内部的生产者-消费者模型，削峰填谷，防止内存无限增长。
 */
template<typename T>
class BoundedQueue {
public:
    explicit BoundedQueue(size_t max_size = 10) : max_size_(max_size) {}
    
    // 禁止拷贝
    BoundedQueue(const BoundedQueue&) = delete;
    BoundedQueue& operator=(const BoundedQueue&) = delete;

    /**
     * @brief 向队列推送数据（阻塞直到有空间或超时）
     * @param item 数据项
     * @param timeout 超时时间
     * @return 是否成功推送
     */
    bool push(const T& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_full_.wait_for(lock, timeout, [this] { return queue_.size() < max_size_ || stopped_; })) {
            return false; // 超时
        }
        if (stopped_) return false;
        queue_.push(item);
        lock.unlock();
        not_empty_.notify_one();
        return true;
    }

    /**
     * @brief 从队列取出数据（阻塞直到有数据或超时）
     * @param item 输出参数
     * @param timeout 超时时间
     * @return 是否成功取出
     */
    bool pop(T& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (!not_empty_.wait_for(lock, timeout, [this] { return !queue_.empty() || stopped_; })) {
            return false;
        }
        if (stopped_ && queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        lock.unlock();
        not_full_.notify_one();
        return true;
    }

    /**
     * @brief 尝试非阻塞弹出
     */
    bool tryPop(T& item) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (queue_.empty() || stopped_) return false;
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }

    /**
     * @brief 停止队列，唤醒所有等待线程
     */
    void stop() {
        stopped_ = true;
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    /**
     * @brief 清空队列
     */
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        std::swap(queue_, empty);
        not_full_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    std::queue<T> queue_;
    size_t max_size_;
    std::atomic<bool> stopped_{false};
};

} // namespace core
} // namespace ai_stream