# VideoMAE 动作识别使用指南

## 概述

本模块使用VideoMAE模型进行视频动作识别，支持以下功能：
- 实时视频流动作检测
- 三分类：攀爬(climb)、打架(fight)、其他(other)
- 帧缓冲和滑动窗口机制
- TensorRT加速推理

## 训练
 run_train.sh
1.conda activate videomae_fune 
2.python3 tools/split_video.py #视频分割为clip
3.python3 tools/prepare_dataset.py #准备数据集
4.python tools/train/videomae_finetune.py \
  --model_name $MODEL_DIR \
  --data_dir $DATA_DIR \
  --num_classes $NUM_CLASSES \
  --output_dir $OUTPUT_DIR \
  --num_epochs $NUM_EPOCHS \
  --batch_size $BATCH_SIZE \
  --fp16 2>&1 | tee $LOG_FILE
5.python3 tools/validate.py #验证

## 模型文件

训练完成后，需要将以下文件复制到项目目录：

```bash
# 创建模型目录
mkdir -p ./models/action_recognition

# 复制TensorRT引擎
cp ./models/videomae/action_recognition.engine ./models/action_recognition/

# 复制类别映射
cp ./models/videomae/finetuned/class_mapping.json ./models/action_recognition/

# 复制配置文件
cp ./config/action_recognition.json ./models/action_recognition/
```

## 配置文件说明

配置文件 `action_recognition.json`：

```json
{
    "model_path": "./models/action_recognition/action_recognition.engine",
    "input_name": "input",
    "input_shape": [1, 16, 3, 224, 224],
    "output_name": "output",
    "num_classes": 3,
    "class_names": ["climb", "fight", "other"],
    "confidence_threshold": 0.7,
    "frame_buffer_size": 16,
    "frame_stride": 8,
    "preprocessing": {
        "image_size": 224,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225]
    }
}
```

### 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| model_path | string | TensorRT引擎路径 |
| num_classes | int | 分类数量（3: climb, fight, other）|
| class_names | array | 类别名称 |
| confidence_threshold | float | 置信度阈值（低于此值不输出）|
| frame_buffer_size | int | 帧缓冲区大小 |
| frame_stride | int | 滑动窗口步长 |

## Pipeline配置

使用pipeline配置文件运行：

```bash
# 运行pipeline
./ai_stream_pipeline --config ./config/pipelines/action_recognition_pipeline.json
```

## 推理流程

```
视频输入 → 帧缓冲 → 滑动窗口采样 → 预处理 → TensorRT推理 → 后处理 → 输出结果
```

### 输入格式
- 形状：`[1, 16, 3, 224, 224]`
- 格式：NCHW (Batch, Frames, Channels, Height, Width)
- 预处理：ImageNet标准化

### 输出格式
- 形状：`[1, 3]`
- 格式：3个类别的softmax概率
- 类别：climb(0), fight(1), other(2)

## C++代码集成

### 基本使用

```cpp
#include "nodes/infer/action_recognition_videomae.h"

// 创建节点
ai_stream::nodes::ActionRecognitionVideoMAENode node;

// 配置参数
ai_stream::nodes::ActionRecognitionVideoMAENode::Config config;
config.model_path = "./models/action_recognition/action_recognition.engine";
config.confidence_threshold = 0.7f;
config.action_labels = {"climb", "fight", "other"};

// 启动节点
node.start();

// 推送帧数据
auto packet = std::make_shared<ai_stream::core::VideoFramePacket>();
packet->stream_id = 1;
packet->mat = cv::make_shared<cv::Mat>(frame);
node.pushData(packet);
```

### 结果处理

```cpp
// 在回调中处理结果
void onActionResult(const ai_stream::core::InferenceResultPacket::ActionResult& result) {
    if (result.confidence >= 0.7f) {
        std::cout << "Action: " << result.action_label 
                  << ", Confidence: " << result.confidence << std::endl;
        
        // 根据动作类型触发告警
        if (result.action_label == "fight") {
            // 触发打架告警
        } else if (result.action_label == "climb") {
            // 触发攀爬告警
        }
    }
}
```

## 性能优化

### 推理延迟
- RTX 3090: ~10ms/帧
- 吞吐量: ~89 QPS

### 优化建议
1. 使用FP16精度（已默认启用）
2. 调整滑动窗口步长（stride）
3. 批量推理（batch_size > 1）

## 故障排除

### 问题1：模型加载失败
```
错误：Failed to open model file
解决：检查model_path是否正确
```

### 问题2：推理结果不准
```
错误：所有类别都输出高置信度
解决：检查预处理是否正确，确保输入帧格式正确
```

### 问题3：GPU内存不足
```
错误：cudaMalloc failed
解决：减小batch_size或使用更小的模型
```

## 类别说明

| 类别 | 标签 | 说明 |
|------|------|------|
| 0 | climb | 攀爬动作 |
| 1 | fight | 打架/冲突动作 |
| 2 | other | 其他动作（行走、跑步等）|

## 注意事项

1. 输入帧必须是RGB格式
2. 帧分辨率会被自动resize到224x224
3. 模型需要16帧作为输入
4. 置信度阈值建议设置为0.7
