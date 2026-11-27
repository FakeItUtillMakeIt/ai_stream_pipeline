#!/usr/bin/env bash
# 启动 HTTP 服务 + Web 前端（+ 内置 mediamtx）
# 镜像 CMD 默认执行本脚本；run-jp72.sh 传入命令时会覆盖。
set -e

HOST="${AI_STREAM_HOST:-0.0.0.0}"
PORT="${AI_STREAM_PORT:-8080}"
MEDIAMTX_CONF="${MEDIAMTX_CONF:-/etc/mediamtx.yml}"

mkdir -p /opt/ai_stream/logs

if command -v mediamtx >/dev/null 2>&1 && [ -f "$MEDIAMTX_CONF" ]; then
    mediamtx "$MEDIAMTX_CONF" >>/opt/ai_stream/logs/mediamtx.log 2>&1 &
    echo "[start-ai-stream] mediamtx started (rtmp:1935 hls:8888)"
    sleep 1
fi

exec /opt/ai_stream/bin/http_server "$HOST" "$PORT" "$@"
