#include <vector>
#include <future>
#include <atomic>
#include <mutex>
#include <queue>
#include <thread>
#include <condition_variable>
#include <functional>

class ThreadPool
{
public:
    ThreadPool(size_t num_threads)
    {
        //LOG_INFO_FMT("Creating thread pool with {} threads", num_threads);

        for (size_t i = 0; i < num_threads; ++i)
        {
            workers_.emplace_back([this]
                                  { worker_thread(); });
        }
    }
    ~ThreadPool()
    {
        {
            std::lock_guard<std::mutex> lock(queue_mutex_);
            stop_ = true;
        }
        condition_.notify_all();

        for (std::thread &worker : workers_)
        {
            if (worker.joinable())
            {
                worker.join();
            }
        }
    }

    template <typename Func, typename... Args>
    auto enqueue(Func &&func, Args &&...args) -> std::future<typename std::result_of<Func(Args...)>::type>
    {
        using return_type = typename std::result_of<Func(Args...)>::type;

        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<Func>(func), std::forward<Args>(args)...));

        std::future<return_type> res = task->get_future();

        {
            std::lock_guard<std::mutex> lock(queue_mutex_);

            if (stop_)
            {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }

            tasks_.emplace([task, this]()
                           {
            active_tasks_++;
            (*task)();
            active_tasks_--;
            completed_tasks_++; });
        }

        condition_.notify_one();
        return res;
    }

    size_t get_queue_size() const
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        return tasks_.size();
    }
    size_t get_active_tasks() const
    {
        return active_tasks_.load();
    }
    size_t get_total_completed_tasks() const
    {
        return completed_tasks_.load();
    }

private:
    void worker_thread()
    {
        while (true)
        {
            std::function<void()> task;

            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                condition_.wait(lock, [this]
                                { return stop_ || !tasks_.empty(); });

                if (stop_ && tasks_.empty())
                {
                    return;
                }

                task = std::move(tasks_.front());
                tasks_.pop();
            }

            try
            {
                task();
            }
            catch (const std::exception &e)
            {
                LOG_ERROR_FMT("Thread pool task failed: {}", e.what());
            }
            catch (...)
            {
                LOG_ERROR("Thread pool task failed with unknown exception");
            }
        }
    }

    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    mutable std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_tasks_{0};
    std::atomic<size_t> completed_tasks_{0};
};
