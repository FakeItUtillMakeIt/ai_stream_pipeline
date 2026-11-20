#!/usr/bin/env bash
# 用 Cloudflare Quick Tunnel 把 Web/登录页 + API + HLS 暴露到外网（单端口）。
#
# 不需要域名、不需要 `cloudflared tunnel login`、不需要 cert.pem：
#   1) 本机启动一个单端口网关，按路径分流到前端、API、HLS；
#   2) cloudflared 把网关端口映射成一个 https://xxx.trycloudflare.com 地址。
#
# 默认直接托管 web/pipeline-editor/dist（静态文件），不依赖 Vite。
# 前端有改动时先执行：cd web/pipeline-editor && npm run build
#
# 用法：
#   ./scripts/expose_quick.sh
#   ./scripts/expose_quick.sh --dev-ui          # 使用 Vite 开发服务器
#   ./scripts/expose_quick.sh --no-tunnel --bind 0.0.0.0
#
# 常用环境变量 / 参数：
#   GATEWAY_PORT   网关端口，默认 8080
#   UI_ROOT        静态前端目录，默认 <repo>/web/pipeline-editor/dist
#   UI_PORT        --dev-ui 时的前端端口，默认 5173，否则 18888
#   APP_PORT       API 端口，默认 18888
#   HLS_PORT       mediamtx HLS 端口，默认 8888
#   HLS_PREFIX     HLS 路径前缀，默认 /live
#   BIND           网关监听地址，默认 127.0.0.1（局域网用 0.0.0.0）
#   --dev-ui       使用 Vite 开发服务器（需要先 npm run dev）
#   --static-ui    强制使用 UI_ROOT 静态目录
#   --no-tunnel    只启动网关，不启动 cloudflared
#
# 启动后：
#   登录页/API : https://<cloudflared 输出的地址>/
#   HLS        : https://<cloudflared 输出的地址>/live/<stream>/index.m3u8
#   把管道 sink 的 play_url 设为上面的 HLS 地址即可。
#
# 注意：quick tunnel 地址每次重启都会变；长期使用请用命名隧道/自有域名。
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GATEWAY_PORT="${GATEWAY_PORT:-8080}"
APP_PORT="${APP_PORT:-18888}"
HLS_PORT="${HLS_PORT:-8888}"
UI_PORT="${UI_PORT:-}"
UI_ROOT="${UI_ROOT:-$REPO_ROOT/web/pipeline-editor/dist}"
HLS_PREFIX="${HLS_PREFIX:-/live}"
BIND="${BIND:-127.0.0.1}"
UI_MODE="${UI_MODE:-auto}"
NO_TUNNEL=0

usage() {
  sed -n '2,35p' "$0" | sed 's/^# \{0,1\}//'
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gateway-port) GATEWAY_PORT="$2"; shift 2 ;;
    --ui-port) UI_PORT="$2"; shift 2 ;;
    --ui-root) UI_ROOT="$2"; shift 2 ;;
    --app-port) APP_PORT="$2"; shift 2 ;;
    --hls-port) HLS_PORT="$2"; shift 2 ;;
    --hls-prefix) HLS_PREFIX="$2"; shift 2 ;;
    --bind) BIND="$2"; shift 2 ;;
    --dev-ui) UI_MODE=dev; shift ;;
    --static-ui) UI_MODE=static; shift ;;
    --no-tunnel) NO_TUNNEL=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "未知参数: $1（用 --help 查看）" >&2; exit 2 ;;
  esac
done

die() { echo "错误: $*" >&2; exit 1; }
warn() { echo "警告: $*" >&2; }

command -v python3 >/dev/null 2>&1 || die "需要 python3"
[[ -f "$SCRIPT_DIR/local_gateway.py" ]] || die "缺少 $SCRIPT_DIR/local_gateway.py"
if [[ "$NO_TUNNEL" != "1" ]]; then
  command -v cloudflared >/dev/null 2>&1 || die "未安装 cloudflared"
fi

if [[ "$UI_MODE" == "auto" ]]; then
  if [[ -f "$UI_ROOT/index.html" ]]; then
    UI_MODE=static
  else
    UI_MODE=dev
  fi
fi

UI_ARGS=()
if [[ "$UI_MODE" == "static" ]]; then
  [[ -f "$UI_ROOT/index.html" ]] || die "静态前端目录缺少 index.html: $UI_ROOT（先 cd web/pipeline-editor && npm run build）"
  UI_ARGS=(--ui-root "$UI_ROOT")
  UI_DESC="static $UI_ROOT"
else
  if [[ -z "$UI_PORT" ]]; then
    if curl -sS -o /dev/null -m 1 "http://127.0.0.1:5173/" 2>/dev/null; then
      UI_PORT=5173
    else
      UI_PORT="$APP_PORT"
    fi
  fi
  UI_ARGS=(--ui-root "")
  UI_DESC="http://127.0.0.1:${UI_PORT}"
  if [[ "$UI_PORT" == "$APP_PORT" ]]; then
    code="$(curl -sS -o /dev/null -w '%{http_code}' -m 2 "http://127.0.0.1:${UI_PORT}/" 2>/dev/null || true)"
    if [[ "$code" == "404" ]]; then
      warn "127.0.0.1:${UI_PORT}/ 返回 404：前端可能没有托管。可设置 AI_STREAM_WEB_ROOT 指向 web/pipeline-editor/dist，或启动 Vite（5173）"
    fi
  fi
fi

if ! curl -fsS -m 2 "http://127.0.0.1:${APP_PORT}/health" >/dev/null 2>&1; then
  warn "无法访问 http://127.0.0.1:${APP_PORT}/health，请确认 ai_stream_server 已启动"
fi
if ! curl -sS -m 2 -o /dev/null "http://127.0.0.1:${HLS_PORT}/" 2>/dev/null; then
  warn "HLS 端口 http://127.0.0.1:${HLS_PORT} 暂不可达（mediamtx 未启动？）"
fi

python3 "$SCRIPT_DIR/local_gateway.py" \
  --listen "${BIND}:${GATEWAY_PORT}" \
  "${UI_ARGS[@]}" \
  --app "127.0.0.1:${APP_PORT}" \
  --hls "127.0.0.1:${HLS_PORT}" \
  --hls-prefix "$HLS_PREFIX" &
GATEWAY_PID=$!

cleanup() {
  kill "$GATEWAY_PID" 2>/dev/null || true
  wait "$GATEWAY_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 30); do
  if ! kill -0 "$GATEWAY_PID" 2>/dev/null; then
    die "网关启动失败"
  fi
  if timeout 1 bash -c "exec 3<>/dev/tcp/127.0.0.1/${GATEWAY_PORT}" 2>/dev/null; then
    break
  fi
  sleep 0.2
done

cat <<EOF
网关已启动: http://${BIND}:${GATEWAY_PORT}/
  前端/登录页 -> ${UI_DESC}
  API/health  -> 127.0.0.1:${APP_PORT}
  HLS ${HLS_PREFIX}/*  -> 127.0.0.1:${HLS_PORT}
EOF

if [[ "$NO_TUNNEL" == "1" ]]; then
  echo "局域网访问: http://<本机IP>:${GATEWAY_PORT}/"
  wait "$GATEWAY_PID"
  exit 0
fi

cat <<EOF

正在启动 Cloudflare Quick Tunnel，稍等它打印公网地址：
  登录页/API : https://<随机>.trycloudflare.com/
  HLS        : https://<随机>.trycloudflare.com${HLS_PREFIX}/<stream>/index.m3u8

把管道 sink 的 play_url 设为上面的 HLS 地址；对外只给实施 token。
按 Ctrl+C 停止隧道和网关。
EOF

cloudflared tunnel --url "http://127.0.0.1:${GATEWAY_PORT}" --no-autoupdate
