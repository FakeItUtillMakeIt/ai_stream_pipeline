#!/bin/bash
# ============================================================================
# Horizon BPU 模型转换脚本
#
# 用法:
#   ./convert.sh <onnx_model> [options]
#
# 示例:
#   ./convert.sh yolov8s.onnx --imgsz 640 --output ./hbm_models
#   ./convert.sh yolov8n.onnx --march nash-m --fast-perf
#   ./convert.sh --config config.yaml
#
# 依赖:
#   pip install horizon_tc_ui  # PTQ 工具链
# ============================================================================

set -e

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_TEMPLATE="${SCRIPT_DIR}/config_template.yaml"

# 默认值
MARCH="nash-m"           # S100P
IMGSZ=640
OUTPUT_DIR="./hbm_output"
FAST_PERF=false
CONFIG_FILE=""
CALIBRATION_DIR=""
CORE_NUM=1
INT16=false              # int16 量化（精度更高，避免分类头塌缩）
FLOAT16=false            # 不做 int8 量化，全 float16（用于先验证输出正确性）
NV12=false               # 运行时输入为 NV12（板端 NV12 直通路径）

# ============================================================================
# 辅助函数
# ============================================================================
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

check_dependencies() {
    # 检查 hb_compile
    if ! command -v hb_compile &> /dev/null; then
        print_error "hb_compile 未安装"
        echo ""
        print_info "Horizon 工具链安装方式:"
        echo "  方式1 (推荐): 使用 Docker"
        echo "    docker pull openexplorer/ai_toolchain_ubuntu_22_s100_s600_cpu:latest"
        echo "    docker run -it --rm -v \$(pwd):/workspace -w /workspace openexplorer/ai_toolchain_ubuntu_22_s100_s600_cpu:latest bash"
        echo ""
        echo "  方式2: 下载 OE 开发包"
        echo "    https://developer.d-robotics.cc/"
        echo "    cd package/host/ai_toolchain && bash install_ai_toolchain.sh"
        echo ""
        exit 1
    fi

    print_info "hb_compile 版本:"
    hb_compile --version 2>/dev/null || echo "  (无法获取版本号)"
}

print_usage() {
    cat << 'EOF'
用法: ./convert.sh <onnx_model> [options]

参数:
  <onnx_model>           ONNX 模型路径

选项:
  --march <arch>         目标平台架构 (默认: nash-m)
                         nash-m: S100P
                         nash-e: S100
                         nash-p: S600
  --imgsz <size>         输入图像尺寸 (默认: 640)
  --output <dir>         输出目录 (默认: ./hbm_output)
  --core-num <n>         BPU 核心数 (默认: 1)
  --calibration <dir>    校准数据目录
  --config <file>        使用自定义配置文件（覆盖其他参数）
  --fast-perf            开启快速性能评测模式
  -h, --help             显示此帮助信息

示例:
  # 转换 YOLOv8s
  ./convert.sh yolov8s.onnx

  # 转换并开启快速性能评测
  ./convert.sh yolov8s.onnx --fast-perf

  # 使用自定义配置
  ./convert.sh --config my_config.yaml

  # 指定校准数据
  ./convert.sh yolov8s.onnx --calibration ./calib_data --imgsz 640
EOF
}

# ============================================================================
# 生成配置文件
# ============================================================================
generate_config() {
    local onnx_model="$1"
    local config_file="$2"

    # 转换为绝对路径
    local onnx_abs
    if [[ "$onnx_model" = /* ]]; then
        onnx_abs="$onnx_model"
    else
        onnx_abs="$(pwd)/$onnx_model"
    fi

    local model_name
    model_name=$(basename "$onnx_model" .onnx)

    local rt_type="rgb"
    [ "$NV12" = true ] && rt_type="nv12"

    cat > "$config_file" << EOF
model_parameters:
  onnx_model: '${onnx_abs}'
  march: '${MARCH}'
  output_model_file_prefix: '${model_name}'

compiler_parameters:
  optimize_level: 'O2'
  core_num: ${CORE_NUM}

input_parameters:
  input_type_rt: '${rt_type}'
  input_type_train: 'rgb'
  input_layout_train: 'NCHW'
  input_batch: 1
  input_shape: '1x3x${IMGSZ}x${IMGSZ}'
  # 校准 npy 为 [0,255] 原值；由工具按 scale_value 归一化到 [0,1]
  # （与官方 RDK 参考一致；norm_type 已弃用，由 mean/scale/std 决定）
  norm_type: 'data_scale'
  scale_value: 0.00392156862745098
EOF
    [ "$NV12" = true ] && print_info "运行时输入类型: nv12（板端可走 NV12 直通路径）"

    # 添加校准参数（如果指定）
    # 注意：HBDK4/hb_compile 的键名是 cal_data_dir，不是 calibration_dir，
    # 写错会导致工具忽略校准并静默走【伪校准】（CALI_TYPE: skip）。
    if [ "$FLOAT16" = true ]; then
        # 不做 int8 量化：全部 float16（无需校准集），用于先验证输出正确性
        cat >> "$config_file" << EOF

calibration_parameters:
  quant_config:
    model_config:
      all_node_type: 'float16'
EOF
        print_info "已启用 float16（不做 int8 量化，无需校准集）"
    elif [ -n "$CALIBRATION_DIR" ]; then
        # 转绝对路径
        local calib_abs
        if [[ "$CALIBRATION_DIR" = /* ]]; then
            calib_abs="$CALIBRATION_DIR"
        else
            calib_abs="$(pwd)/$CALIBRATION_DIR"
        fi
        if [ ! -d "$calib_abs" ]; then
            print_error "校准目录不存在: $calib_abs"
            exit 1
        fi

        # 新版 hb_compile 要求校准数据为 .npy（cal_data_type 已弃用）。
        # 若给的是图片目录，则用与推理一致的预处理生成 npy（每次重新生成，避免复用旧数据）
        local npy_dir="$calib_abs"
        local n_npy n_img
        n_npy=$(find "$calib_abs" -maxdepth 1 -type f -iname '*.npy' | wc -l)
        n_img=$(find "$calib_abs" -maxdepth 1 -type f \( -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.png' -o -iname '*.bmp' \) | wc -l)
        if [ "$n_img" -gt 0 ]; then
            npy_dir="${calib_abs%/}_npy"
            print_info "将 ${n_img} 张图片转换为 npy (letterbox ${IMGSZ}x${IMGSZ}, RGB[0,255] NCHW)..."
            rm -rf "$npy_dir"
            python3 "${SCRIPT_DIR}/make_calib_npy.py" "$calib_abs" "$npy_dir" "$IMGSZ" || {
                print_error "生成 npy 校准数据失败"; exit 1; }
            n_npy=$(find "$npy_dir" -maxdepth 1 -type f -iname '*.npy' | wc -l)
        fi
        if [ "$n_npy" -eq 0 ]; then
            print_error "校准目录内既没有图片也没有 .npy: $calib_abs"
            exit 1
        fi
        if [ "$n_npy" -lt 20 ]; then
            print_warn "校准数据仅 ${n_npy} 个，建议 100~500（至少 20~30）"
        fi
        print_info "使用校准数据: ${npy_dir} (${n_npy} 个 npy)"
        cat >> "$config_file" << EOF

calibration_parameters:
  cal_data_dir: '${npy_dir}'
EOF
        if [ "$INT16" = true ]; then
            cat >> "$config_file" << EOF
  quant_config:
    model_config:
      all_node_type: 'int16'
EOF
            print_info "已启用 int16 量化 (quant_config.model_config.all_node_type=int16)"
        fi
    else
        print_warn "未指定 --calibration：hb_compile 将进行【伪校准】，"
        print_warn "产出的 hbm 没有精度信息，检测/分类结果会退化（仅供功能测试）！"
        print_warn "请先采集校准集: ./capture_calibration.sh <rtsp_url> ./calibration_data 300 2"
    fi

    print_info "配置文件已生成: ${config_file}"
}

# ============================================================================
# 执行转换
# ============================================================================
run_conversion() {
    local config_file="$1"

    print_info "开始模型转换..."
    echo ""

    if [ "$FAST_PERF" = true ]; then
        print_info "模式: 快速性能评测 (--fast-perf)"
        # 从配置文件中提取 onnx_model 和 march
        local onnx_model march
        onnx_model=$(grep "onnx_model:" "$config_file" | awk -F"'" '{print $2}')
        march=$(grep "march:" "$config_file" | awk -F"'" '{print $2}')
        hb_compile --config "$config_file" --fast-perf -m "$onnx_model" --march "$march"
    else
        print_info "模式: 传统转换"
        hb_compile --config "$config_file"
    fi

    local exit_code=$?

    if [ $exit_code -eq 0 ]; then
        echo ""
        print_info "转换成功!"
        print_info "输出目录: ${OUTPUT_DIR}"
        echo ""
        ls -lh "${OUTPUT_DIR}"/*.hbm 2>/dev/null || true
    else
        echo ""
        print_error "转换失败 (exit code: $exit_code)"
        exit $exit_code
    fi
}

# ============================================================================
# 主函数
# ============================================================================
main() {
    local onnx_model=""
    local use_config=false

    # 解析参数
    while [[ $# -gt 0 ]]; do
        case $1 in
            --march)
                MARCH="$2"
                shift 2
                ;;
            --imgsz)
                IMGSZ="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --core-num)
                CORE_NUM="$2"
                shift 2
                ;;
            --calibration)
                CALIBRATION_DIR="$2"
                shift 2
                ;;
            --int16)
                INT16=true
                shift
                ;;
            --float16)
                FLOAT16=true
                shift
                ;;
            --nv12)
                NV12=true
                shift
                ;;
            --config)
                CONFIG_FILE="$2"
                use_config=true
                shift 2
                ;;
            --fast-perf)
                FAST_PERF=true
                shift
                ;;
            -h|--help)
                print_usage
                exit 0
                ;;
            -*)
                print_error "未知选项: $1"
                print_usage
                exit 1
                ;;
            *)
                if [ -z "$onnx_model" ]; then
                    onnx_model="$1"
                else
                    print_error "多余参数: $1"
                    print_usage
                    exit 1
                fi
                shift
                ;;
        esac
    done

    echo "=========================================="
    echo "  Horizon BPU 模型转换工具"
    echo "=========================================="
    echo ""

    # 检查依赖
    check_dependencies
    echo ""

    # 创建输出目录
    mkdir -p "$OUTPUT_DIR"

    # 使用自定义配置或生成配置
    if [ "$use_config" = true ]; then
        if [ ! -f "$CONFIG_FILE" ]; then
            print_error "配置文件不存在: $CONFIG_FILE"
            exit 1
        fi
        print_info "使用配置文件: $CONFIG_FILE"
    else
        # 检查 ONNX 模型
        if [ -z "$onnx_model" ]; then
            print_error "请指定 ONNX 模型路径"
            echo ""
            print_usage
            exit 1
        fi

        if [ ! -f "$onnx_model" ]; then
            print_error "ONNX 模型不存在: $onnx_model"
            exit 1
        fi

        # 生成配置文件
        CONFIG_FILE="$(pwd)/${OUTPUT_DIR}/convert_config.yaml"
        generate_config "$onnx_model" "$CONFIG_FILE"
    fi

    echo ""
    print_info "参数:"
    print_info "  模型: ${onnx_model:-$(grep 'onnx_model' "$CONFIG_FILE" | awk -F"'" '{print $2}')}"
    print_info "  平台: ${MARCH}"
    print_info "  输出: ${OUTPUT_DIR}"
    print_info "  核心数: ${CORE_NUM}"
    print_info "  快速性能: ${FAST_PERF}"
    echo ""

    # 执行转换
    run_conversion "$CONFIG_FILE"
}

main "$@"
