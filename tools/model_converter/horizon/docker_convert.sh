#!/bin/bash
# ============================================================================
# Docker 环境转换脚本
# 自动启动 Horizon Docker 容器并执行模型转换
#
# 用法:
#   ./docker_convert.sh <onnx_model> [options]
#
# 示例:
#   ./docker_convert.sh yolov8s.onnx
#   ./docker_convert.sh yolov8s.onnx --fast-perf
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# Docker 镜像配置
DOCKER_IMAGE="registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0"
DOCKER_IMAGE_GPU="registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0"

# 颜色输出
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

print_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
print_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
print_error() { echo -e "${RED}[ERROR]${NC} $1"; }

print_usage() {
    cat << 'EOF'
用法: ./docker_convert.sh <onnx_model> [options]

参数:
  <onnx_model>           ONNX 模型路径

选项:
  --imgsz <size>         输入图像尺寸 (默认: 640)
  --march <arch>         目标平台 (默认: nash-m/S100P)
  --output <dir>         输出目录 (默认: ./hbm_output)
  --gpu                  使用 GPU Docker 镜像
  --fast-perf            开启快速性能评测
  --pull                 强制拉取最新 Docker 镜像
  -h, --help             显示帮助

示例:
  ./docker_convert.sh yolov8s.onnx
  ./docker_convert.sh yolov8s.onnx --gpu --fast-perf
EOF
}

# 检查 Docker
check_docker() {
    if ! command -v docker &> /dev/null; then
        print_error "Docker 未安装"
        print_info "请先安装 Docker: https://docs.docker.com/install/"
        exit 1
    fi

    if ! docker info &> /dev/null 2>&1; then
        print_error "Docker 服务未运行或当前用户无权限"
        print_info "请运行: sudo systemctl start docker"
        print_info "或添加用户到 docker 组: sudo usermod -aG docker $USER"
        exit 1
    fi
}

# 拉取镜像
pull_image() {
    local image="$1"
    if ! docker image inspect "$image" &> /dev/null 2>&1; then
        print_info "拉取 Docker 镜像: $image"
        docker pull "$image"
    fi
}

# 启动容器执行转换
run_in_docker() {
    local image="$1"
    local onnx_model="$2"
    local output_dir="$3"
    local extra_args="$4"

    # 获取 ONNX 模型的绝对路径
    local onnx_abs
    onnx_abs=$(cd "$(dirname "$onnx_model")" && pwd)/$(basename "$onnx_model")

    # 创建输出目录
    mkdir -p "$output_dir"

    print_info "启动 Docker 容器..."
    print_info "  镜像: $image"
    print_info "  模型: $onnx_abs"
    print_info "  输出: $(pwd)/$output_dir"
    echo ""

    docker run --rm \
        --network host \
        -v "$PROJECT_ROOT":/workspace \
        -v "$onnx_abs":/input/$(basename "$onnx_model") \
        -v "$(pwd)/$output_dir":/output \
        -w /workspace \
        "$image" \
        bash -c "
            cd /workspace/tools/model_converter/horizon
            ./convert.sh /input/$(basename "$onnx_model") \
                --output /output \
                $extra_args
        "

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        print_info "转换完成! 输出目录: $output_dir"
        ls -lh "$output_dir"/*.hbm 2>/dev/null || true
    else
        echo ""
        print_error "转换失败 (exit code: $exit_code)"
        exit $exit_code
    fi
}

main() {
    local onnx_model=""
    local output_dir="./hbm_output"
    local imgsz=640
    local march="nash-m"
    local use_gpu=false
    local fast_perf=false
    local pull=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            --imgsz) IMSGSZ="$2"; shift 2 ;;
            --march) MARCH="$2"; shift 2 ;;
            --output) output_dir="$2"; shift 2 ;;
            --gpu) use_gpu=true; shift ;;
            --fast-perf) fast_perf=true; shift ;;
            --pull) pull=true; shift ;;
            -h|--help) print_usage; exit 0 ;;
            *) onnx_model="$1"; shift ;;
        esac
    done

    echo "=========================================="
    echo "  Horizon BPU Docker 转换工具"
    echo "=========================================="
    echo ""

    # 检查参数
    if [ -z "$onnx_model" ]; then
        print_error "请指定 ONNX 模型路径"
        print_usage
        exit 1
    fi

    if [ ! -f "$onnx_model" ]; then
        print_error "ONNX 模型不存在: $onnx_model"
        exit 1
    fi

    # 检查 Docker
    check_docker

    # 选择镜像
    local image="$DOCKER_IMAGE"
    if [ "$use_gpu" = true ]; then
        image="$DOCKER_IMAGE_GPU"
    fi

    # 拉取镜像
    if [ "$pull" = true ]; then
        print_info "拉取最新镜像: $image"
        docker pull "$image"
    else
        pull_image "$image"
    fi

    # 构建转换参数
    local extra_args="--imgsz $imgsz --march $march"
    if [ "$fast_perf" = true ]; then
        extra_args="$extra_args --fast-perf"
    fi

    # 执行转换
    run_in_docker "$image" "$onnx_model" "$output_dir" "$extra_args"
}

main "$@"
