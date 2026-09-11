#!/bin/bash
# 监控进程 RSS 内存增长（每30s采样一次，持续10分钟）
# 用法：bash monitor_rss.sh <PID> [时长秒数]
PID=${1:-$(pgrep -f simple_detection | head -1)}
DURATION=${2:-600}
INTERVAL=30

if [ -z "$PID" ]; then
    echo "用法: $0 <PID> [时长秒数]"
    exit 1
fi

echo "监控进程 $PID，每${INTERVAL}s采样，共${DURATION}s"
printf "%-8s %-10s %-10s %-10s\n" "时间" "RSS(MB)" "USS(MB)" "PSS(MB)"

for i in $(seq 1 $((DURATION / INTERVAL))); do
    TS=$(date +%H:%M:%S)
    RSS=$(awk '/VmRSS/{printf "%.1f", $2/1024}' /proc/$PID/status 2>/dev/null)
    USS=$(awk '/VmRSS/{print $2}' /proc/$PID/smaps_rollup 2>/dev/null)
    PSS=$(awk '/Pss:/{sum+=$2} END{printf "%.1f", sum/1024}' /proc/$PID/smaps_rollup 2>/dev/null)
    [ -z "$RSS" ] && echo "进程已退出" && break
    printf "%-8s %-10s %-10s %-10s\n" "$TS" "${RSS}M" "${USS:-?}M" "${PSS:-?}M"
    sleep $INTERVAL
done
