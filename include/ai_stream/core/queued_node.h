// include/ai_stream/core/queued_node.h
#pragma once

#include "node.h"
#include "bounded_queue.h"
#include <algorithm>
#include <thread>
#include <chrono>

namespace ai_stream {
namespace core {

/**
 * @brief 带输入队列的异步节点混入基类（背压支持）
 *
 * 用法：将原本的 `class Foo : IXxxNode` 改为 `class Foo : QueuedNode<IXxxNode>`，
 * 并把原 pushData/start/stop 逻辑分别迁移到 processPacket/onStartup/onShutdown。
 *
 * pushData() 只负责入队并立即返回，由专属 worker 线程串行调用
 * processPacket() 处理数据，避免重处理逻辑阻塞上游节点。
 *
 * 队列容量与满队列丢弃策略可通过 JSON 可选字段 "queue" 配置：
 *   "queue": {
 *       "capacity": 64,
 *       "drop_policy": "drop_newest" | "drop_oldest" | "block",
 *       "push_timeout_ms": 10
 *   }
 *
 * @tparam Base 节点接口类型（必须最终继承自 core::Node，如 IDecodeNode）
 */
template<typename Base>
class QueuedNode : public Base {
public:
    enum class DropPolicy {
        DROP_NEWEST,  // 队列满时丢弃新包（默认，直播场景）
        DROP_OLDEST,  // 队列满时丢弃最旧包，新包入队
        BLOCK         // 队列满时阻塞等待 push_timeout_ms，超时丢弃
    };

    explicit QueuedNode(const std::string& name = "QueuedNode") : Base(name) {}

    ~QueuedNode() override {
        QueuedNode::stop();
        if (worker_.joinable()) {
            worker_.join();
        }
    }

    bool configure(const std::string& node_id, const nlohmann::json& params) final {
        if (params.contains("queue") && params["queue"].is_object()) {
            const auto& q = params["queue"];
            queue_capacity_ = static_cast<size_t>(std::max(1, q.value("capacity", static_cast<int>(queue_capacity_))));
            push_timeout_ms_ = q.value("push_timeout_ms", push_timeout_ms_);
            std::string policy = q.value("drop_policy", std::string("drop_newest"));
            if (policy == "drop_oldest") {
                drop_policy_ = DropPolicy::DROP_OLDEST;
            } else if (policy == "block") {
                drop_policy_ = DropPolicy::BLOCK;
            } else {
                drop_policy_ = DropPolicy::DROP_NEWEST;
            }
        }
        return configureImpl(node_id, params);
    }

    bool start() final {
        if (this->running_.load()) return true;
        if (!onStartup()) return false;
        queue_.setMaxSize(queue_capacity_);
        queue_.reset();
        this->running_ = true;
        worker_ = std::thread(&QueuedNode::workerLoop, this);
        return true;
    }

    void stop() final {
        if (!this->running_.exchange(false)) return;
        queue_.stop();
        // stop() 可能由 worker 线程内的 processPacket 触发（如 STREAM_END），
        // 此时不能 join 自身线程，join 交由析构完成
        if (worker_.joinable() && std::this_thread::get_id() != worker_.get_id()) {
            worker_.join();
        }
        onShutdown();
    }

    void pushData(std::shared_ptr<BasePacket> packet) final {
        if (!packet || !this->running_.load()) return;
        bool ok = false;
        switch (drop_policy_) {
            case DropPolicy::DROP_NEWEST:
                ok = queue_.tryPush(packet);
                break;
            case DropPolicy::DROP_OLDEST:
                ok = queue_.tryPush(packet);
                while (!ok) {
                    std::shared_ptr<BasePacket> discarded;
                    if (!queue_.tryPop(discarded)) break;
                    this->recordDropped();
                    ok = queue_.tryPush(packet);
                }
                break;
            case DropPolicy::BLOCK:
                ok = queue_.push(packet, std::chrono::milliseconds(push_timeout_ms_));
                break;
        }
        if (!ok) {
            this->recordDropped();
        }
    }

protected:
    // 派生类的实际处理逻辑，在 worker 线程中串行执行（原 pushData 逻辑迁移至此）
    virtual void processPacket(std::shared_ptr<BasePacket> packet) = 0;

    // 队列空闲（pop 超时）时回调，用于周期性维护任务（如 fusion 的超时合并）
    virtual void onIdle() {}

    // 生命周期钩子：资源初始化/释放（对应原 start/stop 中的非线程逻辑）
    virtual bool onStartup() { return true; }
    virtual void onShutdown() {}

    // 配置解析钩子：默认转发给 Base::configure（接口层的通用配置解析）
    virtual bool configureImpl(const std::string& node_id, const nlohmann::json& params) {
        return Base::configure(node_id, params);
    }

private:
    void workerLoop() {
        while (this->running_.load()) {
            std::shared_ptr<BasePacket> packet;
            if (!queue_.pop(packet, std::chrono::milliseconds(100))) {
                onIdle();
                continue;
            }
            this->in_time_ms_ = utils::TimeUtil::currentTimeMs();
            processPacket(std::move(packet));
        }
        queue_.clear();
    }

    BoundedQueue<std::shared_ptr<BasePacket>> queue_{64};
    std::thread worker_;
    size_t queue_capacity_ = 64;
    DropPolicy drop_policy_ = DropPolicy::DROP_NEWEST;
    int push_timeout_ms_ = 10;
};

} // namespace core
} // namespace ai_stream
