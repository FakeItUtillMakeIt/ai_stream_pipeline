#pragma once

#include <cctype>
#include <filesystem>
#include <string>

namespace ai_stream {
namespace utils {

// 将 pipeline id 转为安全的单级目录名（防路径穿越/非法字符）
inline std::string sanitizePathComponent(const std::string& value) {
    std::string out;
    out.reserve(value.size());
    for (unsigned char c : value) {
        if (std::isalnum(c) || c == '_' || c == '-' || c == '.') {
            out.push_back(static_cast<char>(c));
        } else {
            out.push_back('_');
        }
    }
    while (!out.empty() && out.front() == '.') {
        out.erase(out.begin());
    }
    if (out.empty()) {
        out = "pipeline";
    }
    return out;
}

// 证据目录按实例隔离：<base>/<pipeline_id>。
// - pipeline_id 为空时保持 base（兼容单管道/离线工具）
// - base 末级已是该 id 时不重复追加（兼容旧配置）
inline std::string evidenceSubdir(const std::string& base, const std::string& pipeline_id) {
    if (base.empty() || pipeline_id.empty()) {
        return base;
    }
    const std::string safe = sanitizePathComponent(pipeline_id);
    std::filesystem::path p(base);
    if (p.filename().string() == safe) {
        return base;
    }
    p /= safe;
    return p.lexically_normal().string();
}

} // namespace utils
} // namespace ai_stream
