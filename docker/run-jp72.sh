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
#   HW_CODEC=1   硬解硬编模式：挂载宿主机 /usr/lib/aarch64-linux-gnu 和 /dev，
#                使用宿主同版本的 L4T/TRT/GStreamer 用户态库（需要 --privileged）
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
IMAGE="${IMAGE:-ai_stream_pipeline:jp72}"
DOCKER_USER="${DOCKER_USER:-$(id -u):$(id -g)}"

GPU_LIB_MOUNT=()
if [ -f /opt/nvidia/l4t-gpu-libs/nvgpu/libcuda.so.1 ]; then
    GPU_LIB_MOUNT=(-v /opt/nvidia/l4t-gpu-libs/nvgpu:/opt/nvidia/l4t-gpu-libs/nvgpu:ro)
fi

GPU_ARGS=()
if docker info -f '{{json .Runtimes}}' 2>/dev/null | grep -q '"nvidia"'; then
    # Jetson 常见：l4t 的 nvidia runtime 不支持 --gpus
    GPU_ARGS=(--runtime nvidia)
else
    GPU_ARGS=(--gpus all)
fi

# 进程 GPU 专用内存（NvMap）统计：把 debugfs 子树只读挂进来，
# 非 root 也能读取 /run/nvmap/iovmm/clients（对齐 jtop 口径）
NVMAP_ARGS=()
if [ -d /sys/kernel/debug/nvmap ]; then
    NVMAP_ARGS=(-v /sys/kernel/debug/nvmap:/run/nvmap:ro)
fi

# 组权限：/dev/nvmap 等属于 video 组；容器内组名可能不全，用宿主 GID 添加
GROUP_ARGS=()
for g in video render; do
    gid=$(getent group "$g" 2>/dev/null | cut -d: -f3 || true)
    [ -n "$gid" ] && GROUP_ARGS+=(--group-add "$gid")
done

ENV_ARGS=(-e HOME=/tmp)
[ -n "${VLM_API_KEY:-}" ] && ENV_ARGS+=(-e "VLM_API_KEY=$VLM_API_KEY")
[ -n "${AI_STREAM_API_TOKEN:-}" ] && ENV_ARGS+=(-e "AI_STREAM_API_TOKEN=$AI_STREAM_API_TOKEN")
[ -n "${AI_STREAM_IMPL_TOKEN:-}" ] && ENV_ARGS+=(-e "AI_STREAM_IMPL_TOKEN=$AI_STREAM_IMPL_TOKEN")
[ -n "${AI_STREAM_STATE_DIR:-}" ] && ENV_ARGS+=(-e "AI_STREAM_STATE_DIR=$AI_STREAM_STATE_DIR")
[ -n "${TZ:-}" ] && ENV_ARGS+=(-e "TZ=$TZ")

# 额外挂载目录（同一路径）。例如视频素材在仓库外：
#   MOUNT_DIR=/home/nvidia/lj docker/run-jp72.sh ...
MOUNT_ARGS=()
[ -n "${MOUNT_DIR:-}" ] && MOUNT_ARGS+=(-v "$MOUNT_DIR:$MOUNT_DIR")

# HW_CODEC=1：使用宿主机用户态库（NVIDIA/L4T/TRT/OpenCV/GStreamer 插件），
# 容器内 nvv4l2 硬解硬编才能和宿主驱动版本完全匹配。
# 代价：与宿主机用户态强耦合，镜像内自带的同名库会被覆盖。
HW_CODEC="${HW_CODEC:-0}"
if [ "$HW_CODEC" = "1" ]; then
    EXTRA_ARGS=(--privileged -v /dev:/dev
                -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro)
    ENV_ARGS+=(-e "LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/opt/ai_stream/lib:/usr/local/cuda/lib64:/opt/nvidia/l4t-gpu-libs/nvgpu")
else
    EXTRA_ARGS=()
fi

# Jetson 编解码设备（存在才挂载；HW_CODEC=1 时已整体挂载 /dev）
DEV_ARGS=()
if [ "$HW_CODEC" != "1" ]; then
    for dev in /dev/v4l2-nvdec /dev/v4l2-nvenc /dev/nvmap; do
        if [ -e "$dev" ]; then
            DEV_ARGS+=(--device "$dev")
        fi
    done
    if [ -d /dev/dma_heap ]; then
        DEV_ARGS+=(--device /dev/dma_heap)
    fi
    if [ -d /dev/dri ]; then
        DEV_ARGS+=(--device /dev/dri)
    fi
fi

# NO_MOUNT=1：不挂载仓库，完全使用镜像内置的 /opt/ai_stream（源码/模型/前端/视频）
if [ "${NO_MOUNT:-0}" = "1" ]; then
    WORKDIR=/opt/ai_stream
    REPO_MOUNT=()
else
    WORKDIR="$REPO_ROOT"
    REPO_MOUNT=(-v "$REPO_ROOT:$REPO_ROOT")
fi

echo "==> 镜像:     $IMAGE"
if [ "${NO_MOUNT:-0}" = "1" ]; then
    echo "==> 挂载:     (NO_MOUNT，使用镜像内置资源)"
else
    echo "==> 挂载:     $REPO_ROOT -> $REPO_ROOT"
fi
echo "==> 用户:     $DOCKER_USER"
[ "${#DEV_ARGS[@]}" -gt 0 ] && echo "==> 设备:     ${DEV_ARGS[*]}"

exec docker run -itd --name ai_stream\
    "${GPU_ARGS[@]}" \
    --net host \
    --ipc host \
    --pid host \
    --user "$DOCKER_USER" \
    "${GROUP_ARGS[@]}" \
    "${GPU_LIB_MOUNT[@]}" \
    "${DEV_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" \
    "${ENV_ARGS[@]}" \
    "${MOUNT_ARGS[@]}" \
    "${NVMAP_ARGS[@]}" \
    "${REPO_MOUNT[@]}" \
    -w "$WORKDIR" \
    "$IMAGE" "$@"
