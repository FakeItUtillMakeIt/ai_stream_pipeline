#!/usr/bin/env bash
# 把运行所需的大文件（模型/mediamtx）暂存到 docker/assets/，供 Dockerfile COPY。
# 这些文件不提交到 git（见 .gitignore），但必须存在于构建机上：
#   - 模型：先 git lfs pull（工作区里必须是真实文件，不是 133 字节指针）
#   - mediamtx：默认从 /home/nvidia/lj/pkg 复制，缺失时按版本下载
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ASSETS="$REPO_ROOT/docker/assets"
MEDIAMTX_VER="${MEDIAMTX_VER:-1.1.0}"

mkdir -p "$ASSETS/models/yoloe" "$ASSETS/models/sevncevision" \
         "$ASSETS/models/relateanything" "$ASSETS/models/action_recognition"

stage() {
    local src="$1" dst="$2"
    if [ ! -f "$src" ]; then
        echo "ERROR: 缺少 $src（模型请先 git lfs pull）" >&2
        return 1
    fi
    local size
    size=$(stat -c%s "$src")
    if [ "$size" -lt 128 ]; then
        echo "ERROR: $src 只有 $size 字节，疑似 Git LFS 指针；请先 git lfs pull" >&2
        return 1
    fi
    mkdir -p "$(dirname "$dst")"
    ln -f "$src" "$dst" 2>/dev/null || cp -f "$src" "$dst"
    echo "  $dst ($size bytes)"
}

echo "==> 暂存模型"
stage "$REPO_ROOT/models/yoloe/yoloe_v8s_child_wall_gate.engine" \
      "$ASSETS/models/yoloe/yoloe_v8s_child_wall_gate.engine"
stage "$REPO_ROOT/models/sevncevision/20260921_sevnce_9_cls_nms_trans_640.engine" \
      "$ASSETS/models/sevncevision/20260921_sevnce_9_cls_nms_trans_640.engine"
stage "$REPO_ROOT/models/relateanything/relateanything.engine" \
      "$ASSETS/models/relateanything/relateanything.engine"
stage "$REPO_ROOT/models/action_recognition/action_recognition.rebuild.engine" \
      "$ASSETS/models/action_recognition/action_recognition.rebuild.engine"

stage "$REPO_ROOT/models/relateanything/predicate_bank.json" \
      "$ASSETS/models/relateanything/predicate_bank.json"
stage "$REPO_ROOT/models/relateanything/relateanything.json" \
      "$ASSETS/models/relateanything/relateanything.json" 2>/dev/null || true
stage "$REPO_ROOT/models/relateanything/calibration.json" \
      "$ASSETS/models/relateanything/calibration.json" 2>/dev/null || true
stage "$REPO_ROOT/models/relateanything/thresholds.json" \
      "$ASSETS/models/relateanything/thresholds.json" 2>/dev/null || true
stage "$REPO_ROOT/models/yoloe/yoloe_v8s_child_wall_gate.names.json" \
      "$ASSETS/models/yoloe/yoloe_v8s_child_wall_gate.names.json" 2>/dev/null || true

echo "==> 测试视频"
if [ ! -f "$ASSETS/child_on_wall.mp4" ]; then
    VIDEO="${TEST_VIDEO:-/home/nvidia/lj/child_on_wall.mp4}"
    if [ -f "$VIDEO" ]; then
        cp -f "$VIDEO" "$ASSETS/child_on_wall.mp4"
    else
        echo "ERROR: 缺少测试视频，请放 docker/assets/child_on_wall.mp4 或设置 TEST_VIDEO" >&2
        exit 1
    fi
fi

echo "==> mediamtx"
if [ ! -x "$ASSETS/mediamtx" ]; then
    if [ -x /home/nvidia/lj/pkg/mediamtx ]; then
        cp -f /home/nvidia/lj/pkg/mediamtx "$ASSETS/mediamtx"
    elif command -v mediamtx >/dev/null 2>&1; then
        cp -f "$(command -v mediamtx)" "$ASSETS/mediamtx"
    else
        url="https://github.com/bluenviron/mediamtx/releases/download/v${MEDIAMTX_VER}/mediamtx_v${MEDIAMTX_VER}_linux_arm64v8.tar.gz"
        echo "  下载 $url"
        tmp=$(mktemp -d)
        curl -fsSL "$url" | tar xz -C "$tmp"
        cp -f "$tmp/mediamtx" "$ASSETS/mediamtx"
        [ -f "$ASSETS/mediamtx.yml" ] || cp -f "$tmp/mediamtx.yml" "$ASSETS/mediamtx.yml"
        rm -rf "$tmp"
    fi
    chmod +x "$ASSETS/mediamtx"
fi

chmod +x "$ASSETS"/*.sh "$ASSETS/mediamtx"
echo "==> 资源已就绪：$ASSETS"
