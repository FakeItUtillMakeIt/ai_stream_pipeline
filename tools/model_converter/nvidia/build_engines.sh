#!/usr/bin/env bash
# ONNX -> TensorRT engine 构建脚本（儿童靠近边界链路）
#
# 用法：
#   bash tools/model_converter/nvidia/build_engines.sh [yoloe|relateanything|all]
#
# 前提：
#   - trtexec 在 PATH（本机 TensorRT 10.x）
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
# 精度：默认 --fp16（本机 RTX 5060 实测关系 ~13ms、yoloe ~17ms）。
# 换目标机或要复现分数，把 --fp16 去掉即 FP32（更慢但数值稳）。
set -euo pipefail

# 相对仓库根定位，脚本可在任意 cwd 调用
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
YOLOE_ONNX="$REPO_ROOT/models/yoloe/yoloe_v8s_child_wall_gate.onnx"
RA_ONNX="$REPO_ROOT/models/relateanything/relateanything.onnx"

# 谓词表大小：改谓词库时同步这里（当前 relsgg-vits16plus = 243，嵌入维 512）
VOCAB=243
DIM=512
MAX_BOXES=32      # 与 models/relateanything/relateanything.json 的 max_boxes 一致
IMG=448

build_yoloe() {
    echo "[YOLOE] $YOLOE_ONNX -> .engine"
    trtexec \
        --onnx="$YOLOE_ONNX" \
        --saveEngine="${YOLOE_ONNX%.onnx}.engine" \
        --fp16 \
        --builderOptimizationLevel=5
}

build_relateanything() {
    echo "[RelateAnything] $RA_ONNX -> .engine"
    # 动态维：boxes 的第 1 维 num_boxes 逐帧变化，给 [2, MAX_BOXES] 区间
    local common=(
        --onnx="$RA_ONNX"
        --saveEngine="${RA_ONNX%.onnx}.engine"
        --fp16
        --builderOptimizationLevel=5
        --minShapes="image:1x3x${IMG}x${IMG},boxes:1x2x4,box_counts:1,W:${VOCAB}x${DIM},alpha:${VOCAB}"
        --optShapes="image:1x3x${IMG}x${IMG},boxes:1x${MAX_BOXES}x4,box_counts:1,W:${VOCAB}x${DIM},alpha:${VOCAB}"
        --maxShapes="image:1x3x${IMG}x${IMG},boxes:1x${MAX_BOXES}x4,box_counts:1,W:${VOCAB}x${DIM},alpha:${VOCAB}"
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