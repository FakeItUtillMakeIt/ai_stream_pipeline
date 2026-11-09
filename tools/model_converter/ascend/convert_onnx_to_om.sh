#!/usr/bin/env bash
# tools/model_converter/ascend/convert_onnx_to_om.sh
# ONNX → Ascend .om（ATC）转换脚本
#
# 用法：
#   convert_onnx_to_om.sh <model.onnx> <input_name> <width> <height> [soc_version] [output_prefix]
# 示例（检测，4D NCHW）：
#   convert_onnx_to_om.sh models/sevncevision/20260921_sevnce_9_cls_nms_trans_640.onnx \
#       images 640 640 Ascend310P3 models/sevncevision/20260921_sevnce_9_cls_nms_trans_640
#
# 示例（VideoMAE 动作识别，5D [1,T,3,H,W]）：
#   INPUT_SHAPE="input:1,16,3,224,224" \
#   convert_onnx_to_om.sh models/action_recognition/action_recognition.onnx \
#       input 224 224 Ascend310P3 models/action_recognition/action_recognition
#
# 可用环境变量：
#   INPUT_SHAPE   完整 shape，默认 "<input>:1,3,<H>,<W>"；5D 时显式传入
#   INPUT_FORMAT  默认 auto：4D 用 NCHW，其它维度不传 --input_format
#
# 说明：
# - 检测模型输出 5 个张量；AscendDetectionEngine 按 det_* 名字匹配。
# - 动作识别模型通常输出 [1, num_classes] logits，引擎内部做 softmax。
# - 运行前请 source CANN 的 set_env.sh，或设置 ASCEND_HOME_PATH。
set -eo pipefail

if [ $# -lt 4 ]; then
    sed -n '2,24p' "$0"
    exit 1
fi

MODEL="$1"
INPUT_NAME="$2"
WIDTH="$3"
HEIGHT="$4"
SOC_VERSION="${5:-Ascend310P3}"
OUT_PREFIX="${6:-${MODEL%.onnx}}"
INPUT_SHAPE="${INPUT_SHAPE:-${INPUT_NAME}:1,3,${HEIGHT},${WIDTH}}"
INPUT_FORMAT="${INPUT_FORMAT:-auto}"

if [ ! -f "${MODEL}" ]; then
    echo "model not found: ${MODEL}" >&2
    exit 1
fi

if [ -z "${ASCEND_HOME_PATH:-}" ] && [ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]; then
    # set_env.sh 在 set -u 下会因未定义变量报错，这里临时关闭
    set +u
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
fi

ATC="${ASCEND_HOME_PATH:-/usr/local/Ascend/cann-8.5.0}/bin/atc"
if [ ! -x "${ATC}" ]; then
    ATC="$(command -v atc || true)"
fi
if [ -z "${ATC}" ] || [ ! -x "${ATC}" ]; then
    echo "atc not found; please source /usr/local/Ascend/ascend-toolkit/set_env.sh" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUT_PREFIX}")"

# 4D 形状默认按 NCHW；5D（VideoMAE）不传 --input_format
FORMAT_ARGS=()
if [ "${INPUT_FORMAT}" = "auto" ]; then
    DIMS="${INPUT_SHAPE#*:}"
    DIM_COUNT="$(awk -F',' '{print NF}' <<< "${DIMS}")"
    if [ "${DIM_COUNT}" = "4" ]; then
        FORMAT_ARGS=(--input_format=NCHW)
    fi
elif [ -n "${INPUT_FORMAT}" ]; then
    FORMAT_ARGS=(--input_format="${INPUT_FORMAT}")
fi

echo "ATC: ${ATC}"
echo "model: ${MODEL}"
echo "input: ${INPUT_SHAPE}  soc: ${SOC_VERSION}"
echo "output: ${OUT_PREFIX}.om"

"${ATC}" \
    --framework=5 \
    --model="${MODEL}" \
    --input_shape="${INPUT_SHAPE}" \
    "${FORMAT_ARGS[@]}" \
    --soc_version="${SOC_VERSION}" \
    --output="${OUT_PREFIX}" \
    --log=error

echo "done: ${OUT_PREFIX}.om"
