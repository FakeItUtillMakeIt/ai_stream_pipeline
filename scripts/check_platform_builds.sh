#!/usr/bin/env bash
# scripts/check_platform_builds.sh
# 本地多平台构建矩阵验证：按环境可用性依次验证各配置能否完成编译
set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_ROOT="${BUILD_ROOT:-$ROOT/build-check}"
FAILED=()

have() { command -v "$1" &>/dev/null; }

run_config() {
    local name="$1"; shift
    echo ""
    echo "======================================================"
    echo "[$name] cmake -B $BUILD_ROOT/$name $*"
    echo "======================================================"
    if ! cmake -B "$BUILD_ROOT/$name" -S "$ROOT" \
            -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTS=OFF "$@"; then
        FAILED+=("$name(configure)")
        return
    fi
    if ! cmake --build "$BUILD_ROOT/$name" -j"$(nproc)"; then
        FAILED+=("$name(build)")
        return
    fi
    echo "[$name] OK"
}

echo "=== ai_stream_pipeline 多平台构建检查 ==="

# 1. 纯 CPU 基线（所有平台必须通过）
run_config "cpu" -DWITH_CUDA=OFF -DWITH_TENSORRT=OFF

# 2. CPU + 关闭可选功能（模拟最小依赖环境）
run_config "cpu-minimal" -DWITH_CUDA=OFF -DWITH_TENSORRT=OFF \
    -DWITH_TRACK=OFF -DWITH_ALERT=OFF -DBUILD_HTTP_SERVER=OFF -DBUILD_TOOLS=OFF

# 3. CUDA（本机有 nvidia-smi 时）
if have nvidia-smi && have nvcc; then
    run_config "cuda" -DWITH_CUDA=ON -DWITH_TENSORRT=OFF
else
    echo "[cuda] 跳过（无 GPU 或 nvcc）"
fi

# 4. CUDA + TensorRT
if have nvidia-smi && have nvcc && [ -n "${TENSORRT_DIR:-}" ]; then
    run_config "tensorrt" -DWITH_CUDA=ON -DWITH_TENSORRT=ON
else
    echo "[tensorrt] 跳过（需 TENSORRT_DIR 指向 TensorRT 安装目录）"
fi

echo ""
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "=== 全部配置通过 ==="
else
    echo "=== 失败配置: ${FAILED[*]} ==="
    exit 1
fi
