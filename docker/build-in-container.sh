#!/usr/bin/env bash
# 在容器内增量编译挂载进来的源码（镜像已带完整工具链和依赖）
#
# 用法（容器内）：
#   build-ai-stream                 # Release，构建到 build/
#   build-ai-stream -DWITH_FTP=OFF  # 追加 CMake 参数
set -euo pipefail

SRC="${SRC:-/src}"
BUILD_DIR="${BUILD_DIR:-$SRC/build}"
INSTALL_PREFIX="${INSTALL_PREFIX:-/opt/ai_stream}"
JOBS="${JOBS:-$(nproc)}"

cd "$SRC"

OPTS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$INSTALL_PREFIX"
    -DCMAKE_INSTALL_RPATH='$ORIGIN;$ORIGIN/../lib'
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH:-87}"
    -DOpenCV_DIR=/usr/lib/aarch64-linux-gnu/cmake/opencv4
    -DWITH_CUDA=ON
    -DWITH_TENSORRT=ON
)

echo "==> cmake -S $SRC -B $BUILD_DIR ${OPTS[*]} $*"
cmake -S "$SRC" -B "$BUILD_DIR" "${OPTS[@]}" "$@"
cmake --build "$BUILD_DIR" -j"$JOBS"
cmake --install "$BUILD_DIR"
mkdir -p "$INSTALL_PREFIX/bin"
cp -f "$BUILD_DIR/examples/simple_detection/simple_detection" "$INSTALL_PREFIX/bin/" 2>/dev/null || true
cp -f "$BUILD_DIR/tools/benchmark/bench" "$INSTALL_PREFIX/bin/" 2>/dev/null || true
echo "==> 完成：$INSTALL_PREFIX/bin/"
