// src/nodes/evidence/frame_buffer.cpp
#include "frame_buffer.h"

namespace ai_stream {
namespace nodes {

FrameBuffer::FrameBuffer(size_t capacity) : capacity_(capacity) {}

void FrameBuffer::push(std::shared_ptr<core::VideoFramePacket> frame) {
    if (!frame || !frame->mat || frame->mat->empty()) return;
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.push_back(std::move(frame));
    while (buffer_.size() > capacity_) {
        buffer_.pop_front();
    }
}

std::deque<std::shared_ptr<core::VideoFramePacket>> FrameBuffer::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_;
}

void FrameBuffer::clear() {
    std::lock_guard<std::mutex> lock(mutex_);
    buffer_.clear();
}

size_t FrameBuffer::size() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.size();
}

bool FrameBuffer::empty() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return buffer_.empty();
}

void FrameBuffer::setCapacity(size_t capacity) {
    std::lock_guard<std::mutex> lock(mutex_);
    capacity_ = capacity;
    while (buffer_.size() > capacity_) {
        buffer_.pop_front();
    }
}

} // namespace nodes
} // namespace ai_stream
