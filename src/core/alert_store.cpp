// src/core/alert_store.cpp
#include "ai_stream/core/alert_store.h"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>

namespace fs = std::filesystem;
using nlohmann::json;

namespace ai_stream {
namespace core {

AlertStore& AlertStore::instance() {
    static AlertStore s;
    return s;
}

std::string AlertStore::makeUid(const std::string& pipeline, const std::string& alert_id,
                                int alert_type, int64_t detect_ms) {
    if (!alert_id.empty()) return pipeline + "#" + alert_id;
    return pipeline + "#" + std::to_string(alert_type) + "@" + std::to_string(detect_ms);
}

void AlertStore::init(const std::string& state_dir) {
    std::lock_guard<std::mutex> lk(m_);
    if (inited_) return;
    std::error_code ec;
    fs::create_directories(state_dir, ec);
    events_path_ = (fs::path(state_dir) / "alerts.jsonl").string();
    proc_path_ = (fs::path(state_dir) / "alert_proc.json").string();
    loadLocked();
    inited_ = true;
}

std::string AlertStore::uidOf(const json& r) {
    if (r.contains("uid") && r["uid"].is_string() && !r["uid"].get<std::string>().empty())
        return r["uid"].get<std::string>();
    // 默认 uid = pipeline + alert_id（同一告警事件 occur/last/end 归并为一行）
    std::string p = r.value("pipeline", "");
    std::string a = r.contains("alert") && r["alert"].is_object() ? r["alert"].value("alert_id", "") : "";
    if (a.empty()) a = std::to_string(r.value("ts", 0));
    return p + "#" + a;
}

void AlertStore::loadLocked() {
    recs_.clear(); proc_.clear();
    std::error_code ec;
    // 处理状态
    if (fs::exists(proc_path_, ec)) {
        std::ifstream in(proc_path_); json j; try { in >> j; if (j.is_object()) for (auto& kv : j.items()) proc_[kv.key()] = kv.value(); } catch (...) {}
    }
    // 事件（逐行 JSONL）
    if (fs::exists(events_path_, ec)) {
        std::ifstream in(events_path_);
        std::string line;
        std::vector<json> tmp;
        while (std::getline(in, line)) {
            if (line.empty()) continue;
            try { json r = json::parse(line); tmp.push_back(std::move(r)); } catch (...) {}
        }
        // 按 uid merge_patch 归并（后出现的行把新字段并入已有记录，支持证据→闸门分阶段）
        std::vector<json> merged;
        for (auto& r : tmp) {
            std::string uid = uidOf(r);
            r["uid"] = uid;
            bool done = false;
            for (auto& m : merged) { if (m.value("uid", "") == uid) { m.merge_patch(r); done = true; break; } }
            if (!done) merged.push_back(r);
        }
        // 新→旧：按 ts 降序
        std::sort(merged.begin(), merged.end(), [](const json& a, const json& b) {
            return a.value("ts", (int64_t)0) > b.value("ts", (int64_t)0);
        });
        if (merged.size() > cap_) merged.resize(cap_);
        // 应用已保存的处理状态（缺省 new）
        for (auto& r : merged) {
            std::string uid = r.value("uid", "");
            auto it = proc_.find(uid);
            r["proc"] = (it != proc_.end() && it->second.contains("proc")) ? it->second["proc"].get<std::string>() : std::string("new");
        }
        recs_ = std::move(merged);
    }
}

void AlertStore::append(json record) {
    std::lock_guard<std::mutex> lk(m_);
    if (!record.contains("ts") || !record["ts"].is_number())
        record["ts"] = (int64_t)0;
    std::string uid = uidOf(record);
    record["uid"] = uid;
    // upsert：命中则 merge_patch 合并（保留已有字段与 proc，支持分阶段补充：证据→闸门）
    bool found = false;
    for (auto& r : recs_) {
        if (r.value("uid", "") == uid) {
            r.merge_patch(record);
            found = true;
            break;
        }
    }
    if (!found) {
        record["proc"] = std::string("new");
        recs_.insert(recs_.begin(), std::move(record));  // 新→旧
        if (recs_.size() > cap_) recs_.resize(cap_);
    }
    // 追加事件行（离线可重放；查询用内存）
    if (!events_path_.empty()) {
        std::error_code ec; fs::create_directories(fs::path(events_path_).parent_path(), ec);
        std::ofstream out(events_path_, std::ios::app);
        if (out) out << recs_.front().dump() << "\n";
    }
}

static bool matchText(const json& r, const std::string& q) {
    if (q.empty()) return true;
    std::string hay;
    auto low = [](std::string s) { for (auto& c : s) c = (char)std::tolower((unsigned char)c); return s; };
    hay += low(r.value("pipeline", "")) + " ";
    hay += low(r.value("description", "")) + " " + low(r.value("type", "")) + " ";
    if (r.contains("alert") && r["alert"].is_object()) {
        const json& a = r["alert"];
        hay += low(a.value("alert_name", "")) + " " + low(a.value("alert_type_name", "")) + " " + low(a.value("description", "")) + " ";
    }
    return hay.find(low(q)) != std::string::npos;
}

json AlertStore::query(const std::string& pipeline, const std::string& type, const std::string& proc,
                       int64_t from_ms, int64_t to_ms, const std::string& q, int limit, int offset,
                       int& total_out) {
    std::lock_guard<std::mutex> lk(m_);
    json items = json::array();
    int total = 0, idx = 0;
    for (const auto& r : recs_) {
        if (!pipeline.empty() && r.value("pipeline", "") != pipeline) continue;
        std::string rtype = r.value("type", "");
        if (rtype.empty() && r.contains("alert") && r["alert"].is_object()) rtype = r["alert"].value("alert_type_name", "");
        if (!type.empty() && rtype != type) continue;
        std::string rp = r.value("proc", "new");
        if (!proc.empty() && proc != "all" && rp != proc) continue;
        int64_t ts = r.value("ts", (int64_t)0);
        if (from_ms > 0 && ts < from_ms) continue;
        if (to_ms > 0 && ts > to_ms) continue;
        if (!matchText(r, q)) continue;
        if (idx >= offset && (limit <= 0 || (int)items.size() < limit)) items.push_back(r);
        idx++; total++;
    }
    total_out = total;
    json out;
    out["total"] = total;
    out["items"] = items;
    return out;
}

bool AlertStore::setProc(const std::string& uid, const std::string& proc, const std::string& note, const std::string& by) {
    std::lock_guard<std::mutex> lk(m_);
    bool found = false;
    for (auto& r : recs_) {
        if (r.value("uid", "") == uid) { r["proc"] = proc; found = true; break; }
    }
    if (!found) return false;
    json p = {{"proc", proc}, {"note", note}, {"by", by}, {"ts", (int64_t)0}};
    proc_[uid] = p;
    persistProcLocked();
    return true;
}

void AlertStore::persistProcLocked() {
    if (proc_path_.empty()) return;
    json j = json::object();
    for (auto& kv : proc_) j[kv.first] = kv.second;
    std::error_code ec; fs::create_directories(fs::path(proc_path_).parent_path(), ec);
    std::ofstream out(proc_path_, std::ios::trunc);
    if (out) out << j.dump(2);
}

json AlertStore::stats(const std::string& pipeline) {
    std::lock_guard<std::mutex> lk(m_);
    json s = {{"new", 0}, {"ack", 0}, {"dismiss", 0}, {"resolve", 0}, {"total", 0}};
    for (const auto& r : recs_) {
        if (!pipeline.empty() && r.value("pipeline", "") != pipeline) continue;
        s["total"] = s["total"].get<int>() + 1;
        std::string p = r.value("proc", "new");
        if (s.contains(p)) s[p] = s[p].get<int>() + 1;
    }
    return s;
}

} // namespace core
} // namespace ai_stream
