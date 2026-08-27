// src/utils/time_util.h
#pragma once

#include <chrono>
#include <ctime>
#include <string>

namespace ai_stream {
namespace utils {

class TimeUtil {
public:
    // 线程安全的本地时间转换（POSIX: localtime_r / Win32: localtime_s）
    static std::tm safeLocaltime(std::time_t time_t) {
        std::tm tm_buf{};
#ifdef _WIN32
        localtime_s(&tm_buf, &time_t);
#else
        localtime_r(&time_t, &tm_buf);
#endif
        return tm_buf;
    }

    // 获取当前时间戳（毫秒）
    static int64_t currentTimeMs() {
        return std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // 获取当前时间戳（微秒）
    static int64_t currentTimeUs() {
        return std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::system_clock::now().time_since_epoch()).count();
    }

    // 格式化时间字符串 "2026-04-10 15:30:45.123"
    static std::string formatTime(int64_t timestamp_ms) {
        auto tp = std::chrono::system_clock::time_point(
            std::chrono::milliseconds(timestamp_ms));
        auto time_t = std::chrono::system_clock::to_time_t(tp);
        auto ms = timestamp_ms % 1000;

        std::tm tm_buf = safeLocaltime(time_t);
        char buffer[64];
        std::strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &tm_buf);
        return std::string(buffer) + "." + std::to_string(ms);
    }
};

} // namespace utils
} // namespace ai_stream