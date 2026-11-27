#!/usr/bin/env bash
# 直接运行内置 demo 管道（不启动 HTTP 服务/前端）
# 用法：run-demo [pipeline.json]
set -e

CONF="${1:-/opt/ai_stream/share/ai_stream_pipeline/config/pipelines/child_intrusion_fighting_fusion.demo.json}"
MEDIAMTX_CONF="${MEDIAMTX_CONF:-/etc/mediamtx.yml}"

mkdir -p /opt/ai_stream/logs

if command -v mediamtx >/dev/null 2>&1 && [ -f "$MEDIAMTX_CONF" ]; then
    mediamtx "$MEDIAMTX_CONF" >>/opt/ai_stream/logs/mediamtx.log 2>&1 &
    echo "[run-demo] mediamtx started (rtmp:1935 hls:8888)"
    sleep 1
fi

exec /opt/ai_stream/bin/simple_detection "$CONF" "${@:2}"
