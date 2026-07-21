// src/nodes/evidence/frame_buffer.h
#pragma once

#include "ai_stream/core/packet.h"
#include <deque>
#include <memory>
#include <mutex>

namespace ai_stream {
namespace nodes {

class FrameBuffer {
public:
    explicit FrameBuffer(size_t capacity = 20);
    ~FrameBuffer() = default;

    void push(std::shared_ptr<core::VideoFramePacket> frame);
    std::deque<std::shared_ptr<core::VideoFramePacket>> snapshot() const;
    void clear();
    size_t size() const;
    bool empty() const;
    void setCapacity(size_t capacity);

private:
    size_t capacity_;
    std::deque<std::shared_ptr<core::VideoFramePacket>> buffer_;
    mutable std::mutex mutex_;
};

} // namespace nodes
} // namespace ai_stream
