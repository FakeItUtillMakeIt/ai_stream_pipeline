// include/ai_stream/hal/gpu_buffer_pool.h
// 设备内存池——按字节大小分桶复用 CUDA 全局内存，避免每帧 cudaMalloc/cudaFree。
//
// acquire() 返回带归还删除器的 shared_ptr<void>：引用计数归零时缓冲区
// 回到空闲列表（超出上限才真正 cudaFree）。把它挂到 packet 的
// d_buf_owner / d_bgr_owner 上，即可让设备内存生命周期与下游
// 队列深度自动对齐，消除"单 buffer 每帧复用被后帧覆盖"的竞态。
//
// 仅 CUDA 构建可用；调用方需在 WITH_CUDA 下使用。
#pragma once

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

class GpuBufferPool {
public:
    using Buffer = std::shared_ptr<void>;

    static GpuBufferPool& instance() {
        static GpuBufferPool pool;
        return pool;
    }

    // 申请至少 bytes 字节的设备缓冲区；失败返回空 shared_ptr。
    Buffer acquire(size_t bytes) {
        if (bytes == 0) return nullptr;

        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = free_blocks_.find(bytes);
            if (it != free_blocks_.end() && !it->second.empty()) {
                void* ptr = it->second.front();
                it->second.pop_front();
                if (it->second.empty()) free_blocks_.erase(it);
                return Buffer(ptr, [bytes](void* p) { GpuBufferPool::instance().release(p, bytes); });
            }
        }

        void* ptr = nullptr;
        if (cudaMalloc(&ptr, bytes) != cudaSuccess) {
            LOG_ERROR_FMT("[GpuBufferPool] cudaMalloc {} bytes failed", bytes);
            return nullptr;
        }
        allocated_count_.fetch_add(1);
        return Buffer(ptr, [bytes](void* p) { GpuBufferPool::instance().release(p, bytes); });
    }

    // 空闲块数（诊断用）
    size_t idleBlockCount() const {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t n = 0;
        for (const auto& [_, q] : free_blocks_) n += q.size();
        return n;
    }

    // 累计分配次数（诊断用）
    size_t totalAllocated() const { return allocated_count_.load(); }

private:
    GpuBufferPool() = default;
    ~GpuBufferPool() {
        // 进程退出时统一释放；此时持有者应已全部析构。
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& [_, q] : free_blocks_) {
            for (void* p : q) cudaFree(p);
        }
    }

    void release(void* ptr, size_t bytes) {
        if (!ptr) return;
        std::lock_guard<std::mutex> lock(mutex_);
        auto& q = free_blocks_[bytes];
        if (q.size() >= kMaxIdlePerSize) {
            cudaFree(ptr);  // 空闲超额，直接归还驱动
            return;
        }
        q.push_back(ptr);
    }

    static constexpr size_t kMaxIdlePerSize = 8;  // 每种尺寸最多缓存的空闲块

    mutable std::mutex mutex_;
    std::unordered_map<size_t, std::deque<void*>> free_blocks_;
    std::atomic<size_t> allocated_count_{0};
};

} // namespace hal
} // namespace ai_stream
