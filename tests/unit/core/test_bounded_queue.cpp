// tests/unit/core/test_bounded_queue.cpp
#include <gtest/gtest.h>
#include "ai_stream/core/bounded_queue.h"
#include <thread>
#include <vector>

using ai_stream::core::BoundedQueue;

TEST(BoundedQueueTest, BasicPushPop) {
    BoundedQueue<int> q(4);
    EXPECT_TRUE(q.push(1));
    EXPECT_TRUE(q.push(2));
    EXPECT_EQ(q.size(), 2u);

    int v = 0;
    EXPECT_TRUE(q.pop(v));
    EXPECT_EQ(v, 1);
    EXPECT_TRUE(q.pop(v));
    EXPECT_EQ(v, 2);
    EXPECT_TRUE(q.empty());
}

TEST(BoundedQueueTest, TryPushFailsWhenFull) {
    BoundedQueue<int> q(2);
    EXPECT_TRUE(q.tryPush(1));
    EXPECT_TRUE(q.tryPush(2));
    EXPECT_FALSE(q.tryPush(3));
    EXPECT_EQ(q.size(), 2u);
}

TEST(BoundedQueueTest, PushTimeoutWhenFull) {
    BoundedQueue<int> q(1);
    EXPECT_TRUE(q.push(1));
    auto start = std::chrono::steady_clock::now();
    EXPECT_FALSE(q.push(2, std::chrono::milliseconds(50)));
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::steady_clock::now() - start).count();
    EXPECT_GE(elapsed, 40);
}

TEST(BoundedQueueTest, PopTimeoutWhenEmpty) {
    BoundedQueue<int> q(4);
    int v = 0;
    EXPECT_FALSE(q.pop(v, std::chrono::milliseconds(20)));
}

TEST(BoundedQueueTest, TryPopFailsWhenEmpty) {
    BoundedQueue<int> q(4);
    int v = 0;
    EXPECT_FALSE(q.tryPop(v));
}

TEST(BoundedQueueTest, StopUnblocksWaitingConsumer) {
    BoundedQueue<int> q(4);
    std::atomic<bool> consumer_returned{false};
    std::thread consumer([&] {
        int v = 0;
        q.pop(v, std::chrono::milliseconds(5000));
        consumer_returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    q.stop();
    consumer.join();
    EXPECT_TRUE(consumer_returned.load());

    int v = 0;
    EXPECT_FALSE(q.pop(v, std::chrono::milliseconds(10)));
    EXPECT_FALSE(q.tryPush(1));
}

TEST(BoundedQueueTest, ResetReenablesAfterStop) {
    BoundedQueue<int> q(4);
    EXPECT_TRUE(q.push(1));
    q.stop();
    q.reset();
    EXPECT_TRUE(q.empty());
    EXPECT_TRUE(q.push(2));
    int v = 0;
    EXPECT_TRUE(q.pop(v));
    EXPECT_EQ(v, 2);
}

TEST(BoundedQueueTest, ClearDropsPendingItems) {
    BoundedQueue<int> q(4);
    q.push(1);
    q.push(2);
    q.clear();
    EXPECT_TRUE(q.empty());
    int v = 0;
    EXPECT_FALSE(q.tryPop(v));
}

TEST(BoundedQueueTest, ProducerConsumerOrdering) {
    BoundedQueue<int> q(8);
    constexpr int kCount = 1000;
    std::vector<int> received;
    received.reserve(kCount);

    std::thread consumer([&] {
        for (int i = 0; i < kCount;) {
            int v = 0;
            if (q.pop(v, std::chrono::milliseconds(100))) {
                received.push_back(v);
                ++i;
            }
        }
    });
    for (int i = 0; i < kCount; ++i) {
        while (!q.tryPush(i)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    consumer.join();

    ASSERT_EQ(received.size(), static_cast<size_t>(kCount));
    for (int i = 0; i < kCount; ++i) {
        EXPECT_EQ(received[i], i);
    }
}
