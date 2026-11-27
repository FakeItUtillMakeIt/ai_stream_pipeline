#!/usr/bin/env bash
# 基于已构建的 dev 镜像产出 slim 运行镜像
#
# 用法：
#   docker/build-jp72-runtime.sh
#   DEV_IMAGE=ai_stream_pipeline:jp72 RUNTIME_IMAGE=my/ai_stream:runtime \
#       docker/build-jp72-runtime.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEV_IMAGE="${DEV_IMAGE:-ai_stream_pipeline:jp72}"
RUNTIME_IMAGE="${RUNTIME_IMAGE:-ai_stream_pipeline:jp72-runtime}"
DOCKERFILE="${DOCKERFILE:-$REPO_ROOT/docker/Dockerfile.jp72-runtime}"

if ! docker image inspect "$DEV_IMAGE" >/dev/null 2>&1; then
    echo "ERROR: 未找到 dev 镜像 $DEV_IMAGE，请先执行 docker/build-jp72.sh" >&2
    exit 1
fi

args=(build -f "$DOCKERFILE" -t "$RUNTIME_IMAGE" --build-arg "BUILDER_IMAGE=$DEV_IMAGE")
args+=("$REPO_ROOT")

echo "==> docker ${args[*]}"
docker "${args[@]}"
echo "==> 运行镜像已生成：$RUNTIME_IMAGE"
