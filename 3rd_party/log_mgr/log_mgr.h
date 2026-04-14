#ifndef LOG_MANAGER_H
#define LOG_MANAGER_H

#include <spdlog/spdlog.h>
#include <spdlog/sinks/rotating_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/async.h>
#include <spdlog/pattern_formatter.h>
#include <spdlog/fmt/fmt.h>  // 添加 fmt 支持

#include <memory>
#include <string>
#include <mutex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <backtrace.h>

namespace spdlog
{
    class logger;
    namespace sinks
    {
        class sink;
    }
}

class LogManager
{
public:
    // 日志级别
    enum class Level
    {
        S_TRACE = 0,
        S_DEBUG,
        S_INFO,
        S_WARN,
        S_ERROR,
        S_CRITICAL,
        S_OFF
    };

    // 日志配置
    struct Config
    {
        std::string log_dir;   // 日志目录
        std ::string log_name; // 日志文件名前缀
        Level level;           // 日志级别
        size_t max_file_size;  // 单个文件最大大小
        size_t max_files;      // 最大文件数
        bool console_output;   // 是否输出到控制台
        bool async_mode;       // 是否异步模式
        size_t queue_size;     // 异步队列大小
        int flush_interval;    // 刷新间隔(秒)
        std ::string pattern;  // 日志格式
        bool include_date_in_name; // 是否在文件名中包含日期

        // 构造函数，提供默认值
        Config() : log_dir("logs"),
                   log_name("app"),
                   level(Level::S_INFO),
                   max_file_size(10 * 1024 * 1024), // 10MB
                   max_files(10),
                   console_output(true),
                   async_mode(false), // 默认使用同步模式，更稳定
                   queue_size(8192),
                   flush_interval(3),
                   pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] [%s:%#] %v"),
                   include_date_in_name(true) // 默认包含日期
        {
        }
    };

    // 获取单例实例
    static LogManager &get_instance();

    // 禁止拷贝和移动
    LogManager(const LogManager &) = delete;
    LogManager &operator=(const LogManager &) = delete;
    LogManager(LogManager &&) = delete;
    LogManager &operator=(LogManager &&) = delete;

    // 初始化日志系统
    bool initialize(const Config &config = Config());

    // 重新配置日志系统
    bool reconfigure(const Config &config);

    // 设置日志级别
    void set_level(Level level);

    // 刷新日志
    void flush();

    // 关闭日志系统
    void shutdown();
    
    // 获取当前日期字符串
    std::string get_current_date_str() const;

    public:
    std::string get_last_date() const { return m_current_date; }
    
    void update_date(const std::string& date) {
        if (date != m_current_date) {
            m_current_date = date;
            // 自动触发日志文件轮转
            reconfigure(m_config);
        }
    }

    // 日志记录接口
    void trace(const std::string &message, const char *file = "", int line = 0);
    void debug(const std::string &message, const char *file = "", int line = 0);
    void info(const std::string &message, const char *file = "", int line = 0);
    void warn(const std::string &message, const char *file = "", int line = 0);
    void error(const std::string &message, const char *file = "", int line = 0);
    void critical(const std::string &message, const char *file = "", int line = 0);

    // 格式化日志模板实现
    template <typename... Args>
    void trace_fmt(const char *file, int line, const char *fmt, Args &&...args)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized && m_logger)
        {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog::level::trace, fmt, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void debug_fmt(const char *file, int line, const char *fmt, Args &&...args)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized && m_logger)
        {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog::level::debug, fmt, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void info_fmt(const char *file, int line, const char *fmt, Args &&...args)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized && m_logger)
        {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog::level::info, fmt, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void warn_fmt(const char *file, int line, const char *fmt, Args &&...args)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized && m_logger)
        {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog::level::warn, fmt, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void error_fmt(const char *file, int line, const char *fmt, Args &&...args)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized && m_logger)
        {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog::level::err, fmt, std::forward<Args>(args)...);
        }
    }

    template <typename... Args>
    void critical_fmt(const char *file, int line, const char *fmt, Args &&...args)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_initialized && m_logger)
        {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog::level::critical, fmt, std::forward<Args>(args)...);
        }
    }

    // 获取当前配置
    Config get_config() const;

    // 获取日志目录
    std::string get_log_dir() const;

    // 检查是否已初始化
    bool is_initialized() const;

    static std::string get_stack_trace(int skip_levels = 1);
    
    // 记录信号和堆栈信息
    void log_signal_with_stack(int signal, const std::string& signal_name);
    
    // 是否启用堆栈跟踪（可在配置中添加）
    void enable_stack_trace(bool enable) { m_enable_stack_trace = enable; }
    bool is_stack_trace_enabled() const { return m_enable_stack_trace; }

private:
    LogManager();
    ~LogManager();

    // 创建日志目录
    bool create_log_directory(const std::string &dir);

    // 获取日志级别对应的spdlog级别
    static spdlog::level::level_enum to_spdlog_level(Level level);
    
    // 生成带日期的日志文件名
    std::string generate_log_file_name() const;
    
    static void signal_handler(int signal);
    
    // 设置信号处理
    bool setup_signal_handlers();

    static std::string get_stack_trace_impl(int skip_levels);
    
    // 内部清理函数，不锁定互斥锁
    void cleanup_internal();
private:
    std::shared_ptr<spdlog::logger> m_logger;
    Config m_config;
    mutable std::mutex m_mutex;
    bool m_initialized;
    bool m_async_mode;
    bool m_thread_pool_initialized;
    std::string m_current_date;
    bool m_enable_stack_trace = true;
    bool m_shutting_down = false;  // 添加关闭标志
};

// 为了方便使用，定义一些宏
#define LOG_TRACE(msg) LogManager::get_instance().trace(msg, __FILE__, __LINE__)
#define LOG_DEBUG(msg) LogManager::get_instance().debug(msg, __FILE__, __LINE__)
#define LOG_INFO(msg) LogManager::get_instance().info(msg, __FILE__, __LINE__)
#define LOG_WARN(msg) LogManager::get_instance().warn(msg, __FILE__, __LINE__)
#define LOG_ERROR(msg) LogManager::get_instance().error(msg, __FILE__, __LINE__)
#define LOG_CRITICAL(msg) LogManager::get_instance().critical(msg, __FILE__, __LINE__)

#define LOG_TRACE_FMT(fmt, ...) LogManager::get_instance().trace_fmt(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG_FMT(fmt, ...) LogManager::get_instance().debug_fmt(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_INFO_FMT(fmt, ...) LogManager::get_instance().info_fmt(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_WARN_FMT(fmt, ...) LogManager::get_instance().warn_fmt(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_ERROR_FMT(fmt, ...) LogManager::get_instance().error_fmt(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
#define LOG_CRITICAL_FMT(fmt, ...) LogManager::get_instance().critical_fmt(__FILE__, __LINE__, fmt, ##__VA_ARGS__)

// 检查日期轮转的宏
#define LOG_CHECK_DATE_ROTATION() \
    do { \
        auto current_date = LogManager::get_instance().get_current_date_str(); \
        if (current_date != LogManager::get_instance().get_last_date()) { \
            LogManager::get_instance().update_date(current_date); \
        } \
    } while(0)

#endif // LOG_MANAGER_H