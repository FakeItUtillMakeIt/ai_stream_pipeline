#include "log_mgr/log_mgr.h"

#include <filesystem>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <iostream>
#include <fstream>
#include <regex>
#include <csignal>
#include <execinfo.h>
#include <dlfcn.h>
#include <cxxabi.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <unistd.h>

namespace fs = std::filesystem;

// 单例实例获取
LogManager &LogManager::get_instance()
{
    static LogManager instance;
    instance.enable_stack_trace(true);
    return instance;
}

LogManager::LogManager()
    : m_initialized(false),
      m_async_mode(true),
      m_thread_pool_initialized(false),
      m_current_date(""),
      m_shutting_down(false)
{
}

LogManager::~LogManager()
{
    // 不要在这里调用 shutdown()，避免重复清理
    // 让程序正常退出时自动清理
}

bool LogManager::initialize(const Config &config)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_initialized)
    {
        // 使用cout，因为logger可能还没初始化
        std::cout << "[WARN] LogManager already initialized" << std::endl;
        return true;
    }

    m_config = config;

    // 更新当前日期
    m_current_date = get_current_date_str();

    setup_signal_handlers();
    try
    {
        // 创建日志目录
        if (!create_log_directory(m_config.log_dir))
        {
            std::cout << "[ERROR] Failed to create log directory: " << m_config.log_dir << std::endl;
            return false;
        }

        // 生成日志文件路径（带日期）
        std::string log_file_path = generate_log_file_name();

        // 创建sinks
        std::vector<spdlog::sink_ptr> sinks;

        // 1. 滚动文件sink
        auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file_path,
            m_config.max_file_size,
            m_config.max_files);

        // 设置sink的格式和级别
        rotating_sink->set_pattern(m_config.pattern);
        rotating_sink->set_level(to_spdlog_level(m_config.level));
        sinks.push_back(rotating_sink);

        // 2. 控制台sink (可选)
        if (m_config.console_output)
        {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern(m_config.pattern);
            console_sink->set_level(to_spdlog_level(m_config.level));
            sinks.push_back(console_sink);
        }

        // 创建logger
        if (m_config.async_mode)
        {
            // 异步模式
            if (!m_thread_pool_initialized)
            {
                try {
                    spdlog::init_thread_pool(m_config.queue_size, 1);
                    m_thread_pool_initialized = true;
                } catch (const spdlog::spdlog_ex &ex) {
                    std::cout << "[WARN] Failed to init thread pool: " << ex.what() 
                              << ", falling back to sync mode" << std::endl;
                    m_config.async_mode = false;
                }
            }

            if (m_config.async_mode && m_thread_pool_initialized) {
                m_logger = std::make_shared<spdlog::async_logger>(
                    "multi_sink",
                    sinks.begin(),
                    sinks.end(),
                    spdlog::thread_pool(),
                    spdlog::async_overflow_policy::block);
                m_async_mode = true;
            } else {
                // 回退到同步模式
                m_logger = std::make_shared<spdlog::logger>("multi_sink",
                                                            sinks.begin(), sinks.end());
                m_async_mode = false;
            }
        }
        else
        {
            // 同步模式
            m_logger = std::make_shared<spdlog::logger>("multi_sink",
                                                        sinks.begin(), sinks.end());
            m_async_mode = false;
        }

        m_logger->set_level(to_spdlog_level(m_config.level));

        // 设置刷新策略
        if (m_config.flush_interval > 0)
        {
            spdlog::flush_every(std::chrono::seconds(m_config.flush_interval));
        }

        // 注册logger
        spdlog::register_logger(m_logger);

        // 设置为默认logger
        spdlog::set_default_logger(m_logger);

        m_initialized = true;

        // 记录初始化成功日志
        m_logger->info("LogManager initialized successfully");
        m_logger->info("Log file: {}", log_file_path);
        m_logger->info("File pattern: {}", m_config.include_date_in_name ? m_config.log_name + ".YYYYMMDD.N.log" : m_config.log_name + ".N.log");
        m_logger->info("Max file size: {} bytes", m_config.max_file_size);
        m_logger->info("Max files: {}", m_config.max_files);
        m_logger->info("Async mode: {}", m_config.async_mode ? "enabled" : "disabled");

        return true;
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        std::cout << "[ERROR] Log initialization failed: " << ex.what() << std::endl;
        return false;
    }
    catch (const std::exception &ex)
    {
        std::cout << "[ERROR] Log initialization failed: " << ex.what() << std::endl;
        return false;
    }
}

bool LogManager::reconfigure(const Config &config)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_initialized)
    {
        return initialize(config);
    }

    try
    {
        // 记录重新配置开始
        m_logger->info("Reconfiguring log system...");

        // 保存旧logger用于刷新和删除
        auto old_logger = m_logger;

        // 先删除旧的logger（如果有）
        if (old_logger)
        {
            try {
                old_logger->flush();
                spdlog::drop("multi_sink");  // 明确删除旧logger
            } catch (...) {
                // 忽略异常，可能logger已经被删除
            }
        }

        // 更新配置
        m_config = config;
        m_current_date = get_current_date_str();

        // 生成新的日志文件路径
        std::string log_file_path = generate_log_file_name();

        // 创建新的sinks
        std::vector<spdlog::sink_ptr> sinks;

        // 1. 滚动文件sink
        auto rotating_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
            log_file_path,
            m_config.max_file_size,
            m_config.max_files);

        rotating_sink->set_pattern(m_config.pattern);
        rotating_sink->set_level(to_spdlog_level(m_config.level));
        sinks.push_back(rotating_sink);

        // 2. 控制台sink (可选)
        if (m_config.console_output)
        {
            auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
            console_sink->set_pattern(m_config.pattern);
            console_sink->set_level(to_spdlog_level(m_config.level));
            sinks.push_back(console_sink);
        }

        // 创建新的logger
        std::shared_ptr<spdlog::logger> new_logger;

        if (m_config.async_mode)
        {
            // 异步模式
            if (!m_thread_pool_initialized)
            {
                try {
                    spdlog::init_thread_pool(m_config.queue_size, 1);
                    m_thread_pool_initialized = true;
                } catch (const spdlog::spdlog_ex &ex) {
                    // 这里不能使用m_logger，因为它可能已经被删除
                    std::cerr << "[WARN] Failed to init thread pool: " << ex.what() 
                              << ", falling back to sync mode" << std::endl;
                    m_config.async_mode = false;
                }
            }

            if (m_config.async_mode && m_thread_pool_initialized) {
                new_logger = std::make_shared<spdlog::async_logger>(
                    "multi_sink",
                    sinks.begin(),
                    sinks.end(),
                    spdlog::thread_pool(),
                    spdlog::async_overflow_policy::block);
            } else {
                // 回退到同步模式
                new_logger = std::make_shared<spdlog::logger>("multi_sink",
                                                              sinks.begin(), sinks.end());
            }
        }
        else
        {
            // 同步模式
            new_logger = std::make_shared<spdlog::logger>("multi_sink",
                                                          sinks.begin(), sinks.end());
        }

        new_logger->set_pattern(m_config.pattern);
        new_logger->set_level(to_spdlog_level(m_config.level));

        // 更新logger
        m_logger = new_logger;
        m_async_mode = m_config.async_mode;

        // 注册新的logger并设置为默认
        spdlog::register_logger(m_logger);
        spdlog::set_default_logger(m_logger);

        // 记录重新配置成功
        m_logger->info("Log system reconfigured successfully");
        m_logger->info("New log file: {}", log_file_path);

        return true;
    }
    catch (const spdlog::spdlog_ex &ex)
    {
        // 异常处理中不能使用m_logger，因为它可能创建失败
        std::cerr << "[ERROR] Log reconfiguration failed: " << ex.what() << std::endl;
        return false;
    }
    catch (const std::exception &ex)
    {
        std::cerr << "[ERROR] Log reconfiguration failed: " << ex.what() << std::endl;
        return false;
    }
}

void LogManager::set_level(Level level)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_initialized || !m_logger)
    {
        return;
    }

    m_config.level = level;
    m_logger->set_level(to_spdlog_level(level));
}

void LogManager::flush()
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_initialized && m_logger)
    {
        try {
            m_logger->flush();
        } catch (...) {
            // 忽略刷新异常
        }
    }
}

void LogManager::cleanup_internal()
{
    if (m_shutting_down) {
        return;
    }
    
    m_shutting_down = true;

    if (m_initialized && m_logger)
    {
        try
        {
            // 先刷新日志
            m_logger->flush();
            
            // 等待一段时间，确保异步日志写入
            if (m_async_mode) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            
            // 移除日志器，但不调用 spdlog::drop_all() 避免影响全局
            try {
                spdlog::drop(m_logger->name());
            } catch (...) {
                // 忽略异常
            }
            
            // 重置指针
            m_logger.reset();
            m_initialized = false;
            
            // 如果是异步模式，清理线程池（让 spdlog 自己管理）
            if (m_async_mode) {
                // spdlog 会在程序退出时自动清理线程池
                m_thread_pool_initialized = false;
            }
        }
        catch (const spdlog::spdlog_ex& ex)
        {
            // 输出到标准错误，因为日志系统可能已经关闭
            std::cerr << "[WARN] Error during log shutdown: " << ex.what() << std::endl;
        }
        catch (...)
        {
            // 忽略所有异常
        }
    }
}

void LogManager::shutdown()
{
    std::lock_guard<std::mutex> lock(m_mutex);
    cleanup_internal();
}

std::string LogManager::get_current_date_str() const
{
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    std::tm tm_buf;
    localtime_r(&time_t, &tm_buf);

    std::stringstream ss;
    ss << std::put_time(&tm_buf, "%Y%m%d");
    return ss.str();
}

// 日志记录实现
void LogManager::trace(const std::string &message, const char *file, int line)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized && m_logger && !m_shutting_down)
    {
        try {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog ::level::trace, message);
        } catch (...) {
            // 忽略记录异常
        }
    }
}

void LogManager::debug(const std::string &message, const char *file, int line)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized && m_logger && !m_shutting_down)
    {
        try {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog ::level::debug, message);
        } catch (...) {
            // 忽略记录异常
        }
    }
}

void LogManager::info(const std::string &message, const char *file, int line)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized && m_logger && !m_shutting_down)
    {
        try {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog ::level::info, message);
        } catch (...) {
            // 忽略记录异常
        }
    }
}

void LogManager::warn(const std::string &message, const char *file, int line)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized && m_logger && !m_shutting_down)
    {
        try {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog ::level::warn, message);
        } catch (...) {
            // 忽略记录异常
        }
    }
}

void LogManager::error(const std::string &message, const char *file, int line)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized && m_logger && !m_shutting_down)
    {
        try {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog ::level::err, message);
        } catch (...) {
            // 忽略记录异常
        }
    }
}

void LogManager::critical(const std::string &message, const char *file, int line)
{
    std::lock_guard<std::mutex> lock(m_mutex);
    if (m_initialized && m_logger && !m_shutting_down)
    {
        try {
            m_logger->log(spdlog::source_loc{file, line, ""},
                          spdlog ::level::critical, message);
        } catch (...) {
            // 忽略记录异常
        }
    }
}

// 其他方法实现
LogManager::Config LogManager::get_config() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_config;
}

std::string LogManager::get_log_dir() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_config.log_dir;
}

bool LogManager::is_initialized() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_initialized && !m_shutting_down;
}

// 私有方法实现
bool LogManager::create_log_directory(const std::string &dir)
{
    try
    {
        fs::path dir_path(dir);
        if (!fs::exists(dir_path))
        {
            return fs::create_directories(dir_path);
        }
        return true;
    }
    catch (const fs::filesystem_error &e)
    {
        std::cout << "[ERROR] Failed to create log directory " << dir << ": " << e.what() << std::endl;
        return false;
    }
}

spdlog::level::level_enum LogManager::to_spdlog_level(Level level)
{
    switch (level)
    {
    case Level::S_TRACE:
        return spdlog::level::trace;
    case Level::S_DEBUG:
        return spdlog::level::debug;
    case Level::S_INFO:
        return spdlog::level::info;
    case Level::S_WARN:
        return spdlog::level::warn;
    case Level::S_ERROR:
        return spdlog::level::err;
    case Level::S_CRITICAL:
        return spdlog::level::critical;
    case Level::S_OFF:
        return spdlog::level::off;
    default:
        return spdlog::level::info;
    }
}

std::string LogManager::generate_log_file_name() const
{
    std::stringstream ss;

    if (m_config.include_date_in_name)
    {
        // 格式：app.YYYYMMDD.log
        ss << m_config.log_dir << "/"
           << m_config.log_name << "."
           << m_current_date << ".log";
    }
    else
    {
        // 格式：app.log
        ss << m_config.log_dir << "/"
           << m_config.log_name << ".log";
    }

    return ss.str();
}

// 修改 signal_handler 函数
void LogManager::signal_handler(int signal)
{
    auto &instance = get_instance();

    // 获取信号名称
    const char *signal_name = "";
    const char *signal_desc = "";
    bool is_fatal = true;

    switch (signal)
    {
    case SIGSEGV:
        signal_name = "SIGSEGV";
        signal_desc = "Segmentation Fault (非法内存访问)";
        break;
    case SIGABRT:
        signal_name = "SIGABRT";
        signal_desc = "Abort (程序异常终止)";
        break;
    case SIGTERM:
        signal_name = "SIGTERM";
        signal_desc = "Termination (终止信号)";
        is_fatal = false;
        break;
    case SIGINT:
        signal_name = "SIGINT";
        signal_desc = "Interrupt (用户中断)";
        is_fatal = false;
        break;
    case SIGILL:
        signal_name = "SIGILL";
        signal_desc = "Illegal Instruction (非法指令)";
        break;
    case SIGFPE:
        signal_name = "SIGFPE";
        signal_desc = "Floating Point Exception (浮点异常)";
        break;
    case SIGBUS:
        signal_name = "SIGBUS";
        signal_desc = "Bus Error (总线错误)";
        break;
    default:
        signal_name = "Unknown";
        signal_desc = "Unknown Signal";
        break;
    }

    // 记录信号和堆栈信息
    std::string full_signal_info = std::string(signal_name) + " - " + signal_desc;
    instance.log_signal_with_stack(signal, full_signal_info);

    // 如果是非致命信号，可以恢复默认处理
    if (!is_fatal)
    {
        std::signal(signal, SIG_DFL);
        return;
    }

    // 等待一段时间确保日志写入
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // 恢复默认信号处理并重新触发
    std::signal(signal, SIG_DFL);
    std::raise(signal);
}

bool LogManager::setup_signal_handlers()
{
    // 设置致命信号处理器
    std::signal(SIGSEGV, signal_handler); // 段错误
    std::signal(SIGABRT, signal_handler); // 异常中止
    std::signal(SIGILL, signal_handler);  // 非法指令
    std::signal(SIGFPE, signal_handler);  // 浮点异常
    std::signal(SIGBUS, signal_handler);  // 总线错误

    // 设置终止信号处理器（可选记录）
    std::signal(SIGTERM, signal_handler); // 终止信号
    //std::signal(SIGINT, signal_handler);  // 中断信号

// 其他可能导致崩溃的信号
#ifdef SIGSYS
    std::signal(SIGSYS, signal_handler); // 非法系统调用
#endif

#ifdef SIGXCPU
    std::signal(SIGXCPU, signal_handler); // CPU时间超限
#endif

#ifdef SIGXFSZ
    std::signal(SIGXFSZ, signal_handler); // 文件大小超限
#endif

    return true;
}

std::string LogManager::get_stack_trace(int skip_levels)
{
    return get_stack_trace_impl(skip_levels + 1); // 额外跳过当前函数
}

static std::string run_addr2line(const char* binary, uintptr_t addr_rel) {
    // addr2line 输出两行：1) 函数名 2) file:line
    // -f 显示函数名，-C demangle，-e 指定二进制
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "addr2line -f -C -e \"%s\" 0x%" PRIxPTR " 2>/dev/null", binary, addr_rel);

    FILE* fp = popen(cmd, "r");
    if (!fp) return std::string();

    char func[512] = {0};
    char fileline[512] = {0};

    if (!fgets(func, sizeof(func), fp)) {
        pclose(fp);
        return std::string();
    }
    if (!fgets(fileline, sizeof(fileline), fp)) {
        pclose(fp);
        return std::string();
    }
    pclose(fp);

    // 去掉换行符
    auto trim_newline = [](char* s) {
        if (!s) return;
        size_t len = strlen(s);
        if (len == 0) return;
        if (s[len-1] == '\n') s[len-1] = '\0';
    };
    trim_newline(func);
    trim_newline(fileline);

    std::string out = std::string(func) + " at " + std::string(fileline);
    return out;
}

std::string LogManager::get_stack_trace_impl(int skip_levels)
{
    const int max_frames = 100;
    void* addrlist[max_frames + 1];

    // 获取当前调用堆栈
    int addrlen = backtrace(addrlist, sizeof(addrlist) / sizeof(void*));

    if (addrlen <= skip_levels) {
        return "<no stack trace available>";
    }

    // 解析堆栈地址为可读字符串（备用）
    char** symbollist = backtrace_symbols(addrlist, addrlen);

    std::stringstream ss;
    ss << "Stack trace (most recent call last):\n";

    for (int i = skip_levels; i < addrlen; i++) {
        char linebuf[1024];
        Dl_info info;

        if (dladdr(addrlist[i], &info) && info.dli_fname) {
            const char* symbol_name = info.dli_sname ? info.dli_sname : "??";
            const char* file_name = info.dli_fname ? info.dli_fname : "??";

            // 计算相对于模块基址的偏移（用于 addr2line）
            uintptr_t addr = (uintptr_t)addrlist[i];
            uintptr_t base = (uintptr_t)info.dli_fbase;
            uintptr_t rel = (base != 0) ? (addr - base) : addr;

            // 尝试使用 addr2line 获取 file:line（需要 debug info）
            std::string src = run_addr2line(file_name, rel);

            // 尝试 demangle C++ 函数名（fallback）
            int status = 0;
            char* demangled = nullptr;
            const char* display_name = symbol_name;
            if (info.dli_sname) {
                demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
                if (status == 0 && demangled) {
                    display_name = demangled;
                }
            }

            if (!src.empty()) {
                // 输出带文件和行号
                snprintf(linebuf, sizeof(linebuf),
                         "#%-2d %p %s (%s+0x%" PRIxPTR ") [%s]\n",
                         i - skip_levels,
                         addrlist[i],
                         display_name,
                         file_name,
                         (uintptr_t)((char*)addrlist[i] - (char*)info.dli_saddr),
                         src.c_str());
            } else {
                // addr2line 失败，回退到不带行号的信息
                snprintf(linebuf, sizeof(linebuf),
                         "#%-2d %p %s (%s+0x%" PRIxPTR ")\n",
                         i - skip_levels,
                         addrlist[i],
                         display_name,
                         file_name,
                         (uintptr_t)((char*)addrlist[i] - (char*)info.dli_saddr));
            }

            if (demangled) {
                free(demangled);
            }
        } else {
            // dladdr 失败，使用 backtrace_symbols 的输出作为备用
            snprintf(linebuf, sizeof(linebuf),
                     "#%-2d %s\n",
                     i - skip_levels,
                     (symbollist && symbollist[i]) ? symbollist[i] : "??");
        }

        ss << linebuf;
    }

    if (symbollist) free(symbollist);
    return ss.str();
}

// 记录信号和堆栈信息
void LogManager::log_signal_with_stack(int signal, const std::string &signal_name)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (!m_initialized || !m_logger || m_shutting_down)
    {
        // 如果日志系统未初始化，输出到 stderr
        std::ostringstream oss;
        oss << "\n========================================\n"
            << "FATAL SIGNAL: " << signal << " (" << signal_name << ")\n"
            << get_stack_trace_impl(1) // 跳过 signal_handler
            << "========================================\n"
            << std::endl;
        std::ofstream error_log("error.log", std::ios::app);
        if (error_log.is_open())
        {
            error_log << oss.str();
            error_log.close();
        }
        else
        {
            std::cerr << oss.str();
        }
        return;
    }

    try
    {
        // 记录信号信息
        std::string msg = "FATAL SIGNAL: " + std::to_string(signal) + " (" + signal_name + ")";

        // 记录到日志
        m_logger->log(spdlog::source_loc{__FILE__, __LINE__, ""},
                      spdlog::level::critical, msg);

        // 如果启用堆栈跟踪，记录堆栈信息
        if (m_enable_stack_trace)
        {
            std::string stack_trace = get_stack_trace_impl(2); 
            m_logger->log(spdlog::source_loc{__FILE__, __LINE__, ""},
                          spdlog::level::critical, "Stack trace:\n" + stack_trace);
        }
        
        // 立即刷新，确保日志写入文件
        try {
            m_logger->flush();
        } catch (...) {
            // 忽略刷新异常
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    catch (...)
    {
        // 如果日志记录失败，输出到 stderr
        std::ostringstream oss;
        oss << "\n========================================\n"
            << "FATAL SIGNAL: " << signal << " (" << signal_name << ")\n"
            << "Log system failed to record the error\n"
            << get_stack_trace_impl(1)
            << "========================================\n"
            << std::endl;
        // 输出到 error.log 备用日志文件
        std::ofstream error_log("error.log", std::ios::app);
        if (error_log.is_open())
        {
            error_log << oss.str();
            error_log.close();
        }
        else
        {
            std::cerr << oss.str();
        }
    }
}