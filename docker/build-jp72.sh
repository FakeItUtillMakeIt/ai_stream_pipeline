#!/usr/bin/env bash
# 构建 ai_stream_pipeline JetPack 7.2 镜像
#
# 用法（任意目录）：
#   docker/build-jp72.sh
#   IMAGE=my/ai_stream:jp72 JOBS=8 docker/build-jp72.sh
#   CMAKE_ARGS="-DWITH_FTP=OFF" docker/build-jp72.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="${IMAGE:-ai_stream_pipeline:jp72}"
DOCKERFILE="${DOCKERFILE:-$REPO_ROOT/docker/Dockerfile.jp72}"

args=(build -f "$DOCKERFILE" -t "$IMAGE")
[ -n "${JOBS:-}" ] && args+=(--build-arg "JOBS=$JOBS")
[ -n "${CMAKE_ARGS:-}" ] && args+=(--build-arg "CMAKE_ARGS=$CMAKE_ARGS")
args+=("$REPO_ROOT")

echo "==> docker ${args[*]}"
docker "${args[@]}"
echo "==> 镜像已生成：$IMAGE"
