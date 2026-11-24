#!/usr/bin/env bash
# 启动 ai_stream_pipeline JetPack 7.2 容器
#
# 用法：
#   docker/run-jp72.sh                                  # 进入 bash
#   docker/run-jp72.sh /opt/ai_stream/bin/http_server 0.0.0.0 8080
#   docker/run-jp72.sh /opt/ai_stream/bin/simple_detection config/pipelines/xxx.json
#
# 环境变量：
#   IMAGE        镜像名，默认 ai_stream_pipeline:jp72
#   REPO_ROOT    要挂载的源码/工作目录，默认本仓库根目录
#                挂载到容器内相同绝对路径，这样管道 JSON 里的绝对模型路径不用改
#   DOCKER_USER  容器内 UID:GID，默认当前用户
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-ai_stream_pipeline:jp72}"
DOCKER_USER="${DOCKER_USER:-$(id -u):$(id -g)}"

GPU_LIB_MOUNT=()
if [ -f /opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1 ]; then
    GPU_LIB_MOUNT=(-v /opt/nvidia/l4t-gpu-libs/nvgpu:/opt/nvidia/l4t-gpu-libs/nvgpu:ro)
fi

# Jetson 编解码设备（存在才挂载）
DEV_ARGS=()
for dev in /dev/v4l2-nvdec /dev/v4l2-nvenc /dev/nvmap; do
    if [ -e "$dev" ]; then
        DEV_ARGS+=(--device "$dev")
    fi
done
if [ -d /dev/dri ]; then
    DEV_ARGS+=(--device /dev/dri)
fi

ENV_ARGS=(-e HOME=/tmp)
[ -n "${VLM_API_KEY:-}" ] && ENV_ARGS+=(-e "VLM_API_KEY=$VLM_API_KEY")
[ -n "${AI_STREAM_API_TOKEN:-}" ] && ENV_ARGS+=(-e "AI_STREAM_API_TOKEN=$AI_STREAM_API_TOKEN")
[ -n "${AI_STREAM_STATE_DIR:-}" ] && ENV_ARGS+=(-e "AI_STREAM_STATE_DIR=$AI_STREAM_STATE_DIR")

if [ "$#" -eq 0 ]; then
    set -- bash
fi

echo "==> 镜像:     $IMAGE"
echo "==> 挂载:     $REPO_ROOT -> $REPO_ROOT"
echo "==> 用户:     $DOCKER_USER"
[ "${#DEV_ARGS[@]}" -gt 0 ] && echo "==> 设备:     ${DEV_ARGS[*]}"

exec docker run --rm -it \
    --gpus all \
    --net host \
    --ipc host \
    --user "$DOCKER_USER" \
    "${GPU_LIB_MOUNT[@]}" \
    "${DEV_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    -v "$REPO_ROOT:$REPO_ROOT" \
    -w "$REPO_ROOT" \
    "$IMAGE" "$@"
