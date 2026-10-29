// src/core/alert_store.cpp
#include "ai_stream/core/alert_store.h"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <ctime>

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
    state_dir_ = state_dir;
    proc_path_ = (fs::path(state_dir) / "alert_proc.json").string();
    if (const char* e = std::getenv("AI_STREAM_ALERT_RETENTION_DAYS")) { int v = std::atoi(e); if (v > 0) retention_days_ = v; }
    if (const char* e = std::getenv("AI_STREAM_ALERT_MAX_FILE_MB")) { int v = std::atoi(e); if (v > 0) max_file_bytes_ = (long)v * 1024 * 1024; }
    if (const char* e = std::getenv("AI_STREAM_ALERT_LOAD_DAYS")) { int v = std::atoi(e); if (v > 0) load_days_ = v; }
    loadLocked();
    pruneOldLocked();
    inited_ = true;
}

std::string AlertStore::todayStr() const {
    std::time_t t = std::time(nullptr);
    std::tm lt{};
    localtime_r(&t, &lt);
    char buf[16]; std::strftime(buf, sizeof(buf), "%Y%m%d", &lt);
    return buf;
}

// 当天事件文件；跨天重置切片，超 max_file_bytes_ 则当天内 .N 递增
std::string AlertStore::currentEventsFileLocked() {
    std::string d = todayStr();
    if (d != cur_date_) { cur_date_ = d; cur_slice_ = 0; }
    auto mk = [&](int slice) -> std::string {
        std::string name = slice == 0 ? ("alerts-" + d + ".jsonl")
                                      : ("alerts-" + d + "." + std::to_string(slice) + ".jsonl");
        return (fs::path(state_dir_) / name).string();
    };
    std::string path = mk(cur_slice_);
    std::error_code ec;
    while (fs::exists(path, ec) && fs::file_size(path, ec) > (std::uintmax_t)max_file_bytes_) {
        cur_slice_++; path = mk(cur_slice_);
    }
    return path;
}

void AlertStore::pruneOldLocked() {
    std::error_code ec;
    std::time_t t = std::time(nullptr) - (std::time_t)retention_days_ * 86400;
    std::tm lt{}; localtime_r(&t, &lt);
    char buf[16]; std::strftime(buf, sizeof(buf), "%Y%m%d", &lt);
    std::string cutoff = buf;
    if (!fs::is_directory(state_dir_, ec)) return;
    for (const auto& e : fs::directory_iterator(state_dir_, ec)) {
        std::string fn = e.path().filename().string();
        if (fn.rfind("alerts-", 0) != 0 || fn.size() < 15) continue;
        std::string date = fn.substr(7, 8);
        if (date < cutoff) fs::remove(e.path(), ec);
    }
}

std::string AlertStore::uidOf(const json& r) {
    if (r.contains("uid") && r["uid"].is_string() && !r["uid"].get<std::string>().empty())
        return r["uid"].get<std::string>();
    std::string p = r.value("pipeline", "");
    std::string a = r.contains("alert") && r["alert"].is_object() ? r["alert"].value("alert_id", "") : "";
    if (a.empty()) a = std::to_string(r.value("ts", 0));
    return p + "#" + a;
}

void AlertStore::loadLocked() {
    recs_.clear(); proc_.clear();
    std::error_code ec;
    if (fs::exists(proc_path_, ec)) {
        std::ifstream in(proc_path_); json j;
        try { in >> j; if (j.is_object()) for (auto& kv : j.items()) proc_[kv.key()] = kv.value(); } catch (...) {}
    }
    // 收集 alerts-YYYYMMDD[.N].jsonl
    std::vector<std::pair<std::string, std::string>> files;  // (date, path) 升序
    if (fs::is_directory(state_dir_, ec)) {
        for (const auto& e : fs::directory_iterator(state_dir_, ec)) {
            std::string fn = e.path().filename().string();
            if (fn.rfind("alerts-", 0) != 0 || fn.find(".jsonl") == std::string::npos) continue;
            if (fn.size() < 15) continue;
            files.push_back({fn.substr(7, 8), e.path().string()});
        }
    }
    std::sort(files.begin(), files.end());
    // 只取最近 load_days_ 个不同日期
    std::vector<std::string> dates;
    for (auto it = files.rbegin(); it != files.rend(); ++it) {
        if (dates.size() >= (size_t)load_days_) break;
        if (std::find(dates.begin(), dates.end(), it->first) == dates.end()) dates.push_back(it->first);
    }
    // 旧→新读入选文件（保证同 uid 后读的行覆盖，merge 正确）
    std::vector<json> tmp;
    for (auto& f : files) {
        if (std::find(dates.begin(), dates.end(), f.first) == dates.end()) continue;
        std::ifstream in(f.second); std::string line;
        while (std::getline(in, line)) { if (line.empty()) continue; try { tmp.push_back(json::parse(line)); } catch (...) {} }
    }
    std::vector<json> merged;
    for (auto& r : tmp) {
        std::string uid = uidOf(r); r["uid"] = uid;
        bool done = false;
        for (auto& m : merged) { if (m.value("uid", "") == uid) { m.merge_patch(r); done = true; break; } }
        if (!done) merged.push_back(r);
    }
    std::sort(merged.begin(), merged.end(), [](const json& a, const json& b) {
        return a.value("ts", (int64_t)0) > b.value("ts", (int64_t)0);
    });
    if (merged.size() > cap_) merged.resize(cap_);
    for (auto& r : merged) {
        std::string uid = r.value("uid", "");
        auto it = proc_.find(uid);
        r["proc"] = (it != proc_.end() && it->second.contains("proc")) ? it->second["proc"].get<std::string>() : std::string("new");
    }
    recs_ = std::move(merged);
    cur_date_ = todayStr(); cur_slice_ = 0;
}

void AlertStore::append(json record) {
    std::lock_guard<std::mutex> lk(m_);
    if (!record.contains("ts") || !record["ts"].is_number()) record["ts"] = (int64_t)0;
    std::string uid = uidOf(record);
    record["uid"] = uid;
    json persisted;
    bool found = false;
    for (auto& r : recs_) {
        if (r.value("uid", "") == uid) { r.merge_patch(record); persisted = r; found = true; break; }
    }
    if (!found) {
        record["proc"] = std::string("new");
        recs_.insert(recs_.begin(), record);
        if (recs_.size() > cap_) recs_.resize(cap_);
        persisted = record;
    }
    std::string ef = currentEventsFileLocked();
    std::error_code ec; fs::create_directories(fs::path(ef).parent_path(), ec);
    std::ofstream out(ef, std::ios::app);
    if (out) out << persisted.dump() << "\n";
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
