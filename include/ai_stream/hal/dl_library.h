// include/ai_stream/hal/dl_library.h
// 统一的动态库加载封装：进程级共享（按库名）、引用计数、线程安全。
// 供 RKNN / Horizon / MPP / RGA 等 dlopen 惰性加载后端复用，替代各自
// 重复且非线程安全的 "static handle + check-then-load" 桩。
#pragma once

#include <dlfcn.h>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include "3rd_party/log_mgr/log_mgr.h"

namespace ai_stream {
namespace hal {

class DlLibrary {
public:
    /**
     * @brief 按名称获取动态库（进程级共享，引用计数）
     *
     * 同名库返回同一实例；所有 shared_ptr 释放后自动 dlclose，下次调用重新加载。
     * 线程安全（内部互斥 + 弱引用注册表）。
     */
    static std::shared_ptr<DlLibrary> get(const std::string& name,
                                          int flags = RTLD_NOW | RTLD_GLOBAL) {
        static std::mutex registry_mutex;
        static std::unordered_map<std::string, std::weak_ptr<DlLibrary>> registry;
        std::lock_guard<std::mutex> lock(registry_mutex);
        auto it = registry.find(name);
        if (it != registry.end()) {
            if (auto sp = it->second.lock()) {
                return sp;
            }
        }
        auto lib = std::shared_ptr<DlLibrary>(new DlLibrary(name, flags));
        registry[name] = lib;
        return lib;
    }

    ~DlLibrary() {
        if (handle_) {
            dlclose(handle_);
            handle_ = nullptr;
        }
    }

    DlLibrary(const DlLibrary&) = delete;
    DlLibrary& operator=(const DlLibrary&) = delete;

    bool isOpen() const { return handle_ != nullptr; }
    const std::string& name() const { return name_; }

    void* sym(const char* symbol) const {
        return handle_ ? dlsym(handle_, symbol) : nullptr;
    }

    template <typename Fn>
    Fn symAs(const char* symbol) const {
        return reinterpret_cast<Fn>(sym(symbol));
    }

private:
    DlLibrary(const std::string& name, int flags) : name_(name) {
        handle_ = dlopen(name.c_str(), flags);
        if (!handle_) {
            LOG_WARN_FMT("[DlLibrary] dlopen {} failed: {}", name, dlerror());
        }
    }

    std::string name_;
    void* handle_ = nullptr;
};

} // namespace hal
} // namespace ai_stream
