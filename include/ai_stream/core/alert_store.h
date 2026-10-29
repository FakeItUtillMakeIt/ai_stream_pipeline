// include/ai_stream/core/alert_store.h
#pragma once
#include <string>
#include <vector>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>

namespace ai_stream {
namespace core {

/**
 * @brief 告警历史存储（单例）
 *  - 由 report_sink 在告警发生时 append（按 uid upsert，含 snapshot/vlm 等完整字段）
 *  - 由 ApiServer 提供查询/确认接口
 *  - 事件持久化到 <state>/alerts.jsonl（追加），处理状态到 <state>/alert_proc.json（覆盖）
 *  - 内存保留最近 cap 条用于快速检索
 */
class AlertStore {
public:
    static AlertStore& instance();

    // 统一 uid 生成：alert_id 非空用 pipeline#alert_id；否则用 pipeline#type#detect_ms
    // （detect_ms 在一次告警存续期稳定，occur/last/end 归并为一行，不同告警分开）
    static std::string makeUid(const std::string& pipeline, const std::string& alert_id,
                               int alert_type, int64_t detect_ms);

    // 设定状态目录并加载历史（幂等；HTTP 服务启动时调用）
    void init(const std::string& state_dir);

    // 追加/更新一条告警（record 需含 uid 或由本函数生成；proc 字段由确认接口维护）
    void append(nlohmann::json record);

    // 查询：按 pipeline/type/proc/时间范围/关键字过滤，返回 {total, items[]}（新→旧）
    nlohmann::json query(const std::string& pipeline, const std::string& type, const std::string& proc,
                         int64_t from_ms, int64_t to_ms, const std::string& q, int limit, int offset,
                         int& total_out);

    // 更新处理状态：proc ∈ new|ack|dismiss|resolve
    bool setProc(const std::string& uid, const std::string& proc, const std::string& note, const std::string& by);

    // 各处理状态计数：{new, ack, dismiss, resolve}
    nlohmann::json stats(const std::string& pipeline);

private:
    AlertStore() = default;
    void loadLocked();
    void persistProcLocked();
    std::string uidOf(const nlohmann::json& r);
    std::string todayStr() const;                 // 本地 YYYYMMDD
    std::string currentEventsFileLocked();        // 当天文件(必要时按量切片 .N)，必要时轮转
    void pruneOldLocked();                        // 删除超过保留天数的旧文件

    std::mutex m_;
    std::vector<nlohmann::json> recs_;          // 新→旧
    std::map<std::string, nlohmann::json> proc_;// uid -> {proc,note,by,ts}
    std::string state_dir_, proc_path_;
    std::string cur_date_;                       // 当前写入的日期串
    int cur_slice_ = 0;                          // 当天第几个切片
    bool inited_ = false;
    size_t cap_ = 500;                          // 内存保留条数
    long max_file_bytes_ = 50L * 1024 * 1024;    // 单文件上限(超出当天内切片)
    int load_days_ = 90;                          // 启动加载最近 N 天（与保留期一致；内存由 cap 兜底）
    int retention_days_ = 90;                    // 保留天数
};

} // namespace core
} // namespace ai_stream
