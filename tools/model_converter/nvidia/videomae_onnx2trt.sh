#!/bin/bash
# VideoMAE ONNX到TensorRT转换脚本

set -e

# 检查参数
if [ $# -lt 1 ]; then
    echo "Usage: $0 <onnx_model> [engine_model]"
    echo "Example: $0 videomae_small.onnx videomae_small.engine"
    exit 1
fi

ONNX_MODEL=$1
ENGINE_MODEL=${2:-${ONNX_MODEL%.onnx}.engine}

echo "Converting ONNX to TensorRT engine..."
echo "Input: $ONNX_MODEL"
echo "Output: $ENGINE_MODEL"

# 检查trtexec是否存在
if ! command -v trtexec &> /dev/null; then
    # 尝试查找TensorRT安装路径
    TRT_EXEC=""
    for path in /usr/src/tensorrt/bin/trtexec \
                /usr/local/tensorrt/bin/trtexec \
                /opt/TensorRT/bin/trtexec \
                $TENSORRT_DIR/bin/trtexec; do
        if [ -f "$path" ]; then
            TRT_EXEC=$path
            break
        fi
    done
    
    if [ -z "$TRT_EXEC" ]; then
        echo "Error: trtexec not found. Please install TensorRT or set TENSORRT_DIR environment variable."
        exit 1
    fi
else
    TRT_EXEC=trtexec
fi

# 使用trtexec转换
# 参数说明：
# --onnx: 输入ONNX模型
# --saveEngine: 输出TensorRT引擎
# --fp16: 启用FP16精度（推荐，可大幅加速）
# --workspace: 工作空间大小（MB）
# --minShapes/--optShapes/--maxShapes: 动态shape配置

$TRT_EXEC \
    --onnx=$ONNX_MODEL \
    --saveEngine=$ENGINE_MODEL \
    --fp16 \
    --workspace=4096 \
    --minShapes=input:1x16x3x224x224 \
    --optShapes=input:1x16x3x224x224 \
    --maxShapes=input:4x16x3x224x224

echo ""
echo "TensorRT engine saved to: $ENGINE_MODEL"
echo "Model conversion completed successfully!"
