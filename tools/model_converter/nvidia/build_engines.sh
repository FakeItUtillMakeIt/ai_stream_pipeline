#!/usr/bin/env bash
# ONNX -> TensorRT engine 构建脚本（儿童靠近边界链路）
#
# 用法：
#   bash tools/model_converter/nvidia/build_engines.sh [yoloe|relateanything|all]
#
# 前提：
#   - trtexec 在 PATH（自动适配 TensorRT 10.x / 11.x，见下方"精度"说明）
#   - 对应 .onnx 已在 models/<dir>/ 下
#
# 为什么要用脚本而不是随手敲 trtexec：
#   1) 引擎绑定 GPU 架构 + TensorRT 版本，必须在**目标机**重建，脚本保证参数一致；
#   2) RelateAnything 的 num_boxes 是动态维，必须给 min/opt/max profile，
#      漏了会在运行期 setInputShape 才炸；box_counts / alpha 是 1-D，
#      写成 2-D（1x1 / 243x1）会被 TRT 直接拒绝（"profile 0 has 2 dimensions"）；
#   3) YOLOE 是端到端 NMS 图，输出维 (N,...) 随检出数变，但 TRT 能自己处理
#      动态维，不需要 profile，只要 batch 固定为 1。
#
# 精度：
#   - TensorRT 10.x：trtexec --fp16 弱类型构建（本机 RTX 5060 实测关系 ~13ms、yoloe ~17ms）。
#   - TensorRT 11.x：精度标志已全部移除（强类型为唯一模式）。需要先用 ModelOpt
#     AutoCast 把 ONNX 离线转成 FP16 混合精度模型，再原样构建：
#       pip install "nvidia-modelopt[onnx]"
#       python -m modelopt.onnx.autocast --onnx_path x.onnx --output_path x.fp16.onnx
#     本脚本检测到 TRT 11 + 已安装 modelopt 时会自动执行该步骤。
#   要复现分数（FP32），设环境变量 PRECISION=fp32。
set -euo pipefail

# 相对仓库根定位，脚本可在任意 cwd 调用
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLOE_ONNX="$REPO_ROOT/models/yoloe/yoloe_v8s_child_wall_gate.onnx"
RA_ONNX="$REPO_ROOT/models/relateanything/relateanything.onnx"

PRECISION="${PRECISION:-fp16}"

# 检测 TensorRT 主版本（trtexec 版本行含 [TensorRT v110000] 标记 → 主版本 11）
TRT_RAW=$( (trtexec --version 2>&1 || true) | grep -oE 'TensorRT v[0-9]+' | grep -oE '[0-9]+' | head -1 )
if [ -n "$TRT_RAW" ]; then
    TRT_MAJOR=$((10#${TRT_RAW:0:2}))
else
    TRT_MAJOR=10
fi

# 若需要 FP16 且是 TRT 11：尝试用 ModelOpt AutoCast 生成混合精度 ONNX
# 成功则返回 .fp16.onnx 路径，失败/未安装则回退 FP32 并告警
autocast_onnx() {
    local onnx="$1"
    local fp16_onnx="${onnx%.onnx}.fp16.onnx"
    if [ -f "$fp16_onnx" ] && [ "$fp16_onnx" -nt "$onnx" ]; then
        echo "$fp16_onnx"; return 0
    fi
    if python3 -c "import modelopt" 2>/dev/null; then
        echo "[AutoCast] $onnx -> $fp16_onnx"
        python3 -m modelopt.onnx.autocast --onnx_path="$onnx" --output_path="$fp16_onnx" >/dev/null
        echo "$fp16_onnx"; return 0
    fi
    echo "[WARN] TRT 11 无 --fp16；未安装 nvidia-modelopt，本模型回退 FP32 构建。" >&2
    echo "       如需 FP16：pip install 'nvidia-modelopt[onnx]' 后重跑本脚本。" >&2
    return 1
}

# 组装精度相关参数到数组 PREC_FLAGS（供 "$@" 展开）
resolve_precision() {
    local onnx="$1"
    PREC_ONNX="$onnx"
    PREC_FLAGS=()
    if [ "$PRECISION" = "fp32" ]; then
        return 0
    fi
    if [ "$TRT_MAJOR" -ge 11 ]; then
        # TRT11：靠 AutoCast 烘焙精度，trtexec 无精度 flag
        if ! PREC_ONNX=$(autocast_onnx "$onnx"); then
            PREC_ONNX="$onnx"   # 回退 FP32
        fi
    else
        PREC_FLAGS+=(--fp16)
    fi
}

# 谓词表大小：改谓词库时同步这里（当前 relsgg-vits16plus = 243，嵌入维 512）
VOCAB=243
DIM=512
MAX_BOXES=32      # 与 models/relateanything/relateanything.json 的 max_boxes 一致
IMG=448

build_yoloe() {
    resolve_precision "$YOLOE_ONNX"
    echo "[YOLOE] $PREC_ONNX -> .engine (TRT ${TRT_MAJOR}, ${PRECISION})"
    trtexec \
        --onnx="$PREC_ONNX" \
        --saveEngine="${YOLOE_ONNX%.onnx}.engine" \
        --builderOptimizationLevel=5 \
        "${PREC_FLAGS[@]}"
}

build_relateanything() {
    resolve_precision "$RA_ONNX"
    echo "[RelateAnything] $PREC_ONNX -> .engine (TRT ${TRT_MAJOR}, ${PRECISION})"
    # 动态维：boxes 的第 1 维 num_boxes 逐帧变化，给 [2, MAX_BOXES] 区间
    local common=(
        --onnx="$PREC_ONNX"
        --saveEngine="${RA_ONNX%.onnx}.engine"
        --builderOptimizationLevel=5
        --minShapes="image:1x3x${IMG}x${IMG},boxes:1x2x4,box_counts:1,W:${VOCAB}x${DIM},alpha:${VOCAB}"
        --optShapes="image:1x3x${IMG}x${IMG},boxes:1x${MAX_BOXES}x4,box_counts:1,W:${VOCAB}x${DIM},alpha:${VOCAB}"
        --maxShapes="image:1x3x${IMG}x${IMG},boxes:1x${MAX_BOXES}x4,box_counts:1,W:${VOCAB}x${DIM},alpha:${VOCAB}"
        "${PREC_FLAGS[@]}"
    )
    trtexec "${common[@]}"
}

case "${1:-all}" in
    yoloe)          build_yoloe ;;
    relateanything) build_relateanything ;;
    all)            build_yoloe; build_relateanything ;;
    *) echo "unknown target: $1 (yoloe|relateanything|all)"; exit 2 ;;
esac

echo "完成。引擎与 .onnx 同目录、同名（.engine 后缀）。"