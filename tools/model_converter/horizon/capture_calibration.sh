#!/bin/bash
# ============================================================================
# 从 RTSP 视频流采集 INT8 量化校准集
#
# 用法:
#   ./capture_calibration.sh <rtsp_url> [out_dir] [count] [fps]
#
# 示例:
#   ./capture_calibration.sh rtsp://localhost:8554/stream1 ./calibration_data 300 2
#
# 说明:
#   - count: 采集张数（建议 200~500）
#   - fps  : 每秒抽帧数（避免相邻帧高度相似，建议 1~5）
#   - 输出为 jpg 原图，交给 hb_compile 按配置自行 resize/归一化
# ============================================================================
set -e

RTSP_URL="${1:-rtsp://localhost:8554/stream1}"
OUT_DIR="${2:-./calibration_data}"
COUNT="${3:-300}"
FPS="${4:-2}"

mkdir -p "$OUT_DIR"

echo "[INFO] RTSP: $RTSP_URL"
echo "[INFO] 输出: $OUT_DIR  (张数=$COUNT, 抽帧=$FPS fps)"
echo "[INFO] 开始采集，请确保画面覆盖不同场景/光照/目标..."

ffmpeg -y -rtsp_transport tcp -i "$RTSP_URL" \
    -vf "fps=${FPS}" -frames:v "${COUNT}" -q:v 2 \
    "$OUT_DIR/calib_%04d.jpg"

N=$(ls -1 "$OUT_DIR"/*.jpg 2>/dev/null | wc -l)
echo "[INFO] 采集完成，共 ${N} 张 -> $OUT_DIR"
