#!/bin/bash
# VideoMAE 训练启动脚本
# 使用方法: bash tools/train/run_train.sh

# 激活 conda 环境
source ~/miniforge3/etc/profile.d/conda.sh
conda activate videomae_fune

# 设置环境变量
export PYTHONUNBUFFERED=1  # 实时打印输出

# 指定 GPU（修改这里选择 GPU 编号）
export CUDA_VISIBLE_DEVICES=0

# 训练参数
MODEL_DIR="./models/videomae/pretrained"
DATA_DIR="/home/sevnce/lj/project/video_dataset"
OUTPUT_DIR="./models/videomae/finetuned"
NUM_CLASSES=3
NUM_EPOCHS=30
BATCH_SIZE=4
LOG_FILE="./models/videomae/train_$(date +%Y%m%d_%H%M%S).log"

# 打印配置
echo "=========================================="
echo "Training Configuration"
echo "=========================================="
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Model: $MODEL_DIR"
echo "Data: $DATA_DIR"
echo "Output: $OUTPUT_DIR"
echo "Classes: $NUM_CLASSES"
echo "Epochs: $NUM_EPOCHS"
echo "Batch Size: $BATCH_SIZE"
echo "Log File: $LOG_FILE"
echo "=========================================="

# 创建输出目录
mkdir -p $(dirname $LOG_FILE)

# 启动训练（带日志输出）
python tools/train/videomae_finetune.py \
  --model_name $MODEL_DIR \
  --data_dir $DATA_DIR \
  --num_classes $NUM_CLASSES \
  --output_dir $OUTPUT_DIR \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --fp16 2>&1 | tee $LOG_FILE

echo ""
echo "Training completed!"
echo "Log saved to: $LOG_FILE"
