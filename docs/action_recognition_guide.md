# VideoMAE 动作识别使用指南

## 概述

本模块使用VideoMAE模型进行视频动作识别，支持以下功能：
- 实时视频流动作检测
- 三分类：攀爬(climb)、打架(fight)、其他(other)
- 帧缓冲和滑动窗口机制
- TensorRT 加速推理（HAL 后端：`src/hal/tensorrt/tensorrt_action_recognition.h`）

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

训练完成后，将导出的 TensorRT 引擎放到项目目录：

```bash
mkdir -p ./models/action_recognition/
cp <训练产物>/action_recognition.engine ./models/action_recognition/
```

引擎转换工具见 `tools/model_converter/`。

## 管道配置（推荐方式）

动作识别节点通过管道 JSON 的节点 `params` 配置，
参考模板 `config/pipelines/action_recongnition_pipeline.json`：

```json
{
  "id": "videomae_test",
  "graph": {
    "nodes": [
      { "id": "src1", "type": "rtsp_source",
        "params": { "url": "rtsp://localhost:8554/stream1", "skip_frames": 1 } },
      { "id": "decode1", "type": "ffmpeg_decode",
        "params": { "output_bgr": true, "hw_decoder": true, "hw_decoder_type": "h264_cuvid" } },
      { "id": "preprocess_videomae", "type": "resize_normalize",
        "params": { "output_width": 224, "output_height": 224,
                    "keep_aspect_ratio": false,
                    "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225] } },
      { "id": "action_recognition1", "type": "action_recognition_videomae",
        "params": {
          "model_path": "./models/action_recognition/action_recognition.engine",
          "confidence_threshold": 0.7,
          "num_frames": 16,
          "frame_interval": 2,
          "window_size": 16,
          "stride": 8,
          "batch_size": 1,
          "input_height": 224,
          "input_width": 224,
          "action_labels": ["climb", "fight", "other"]
        } }
    ],
    "edges": [
      { "from": "src1", "to": "decode1" },
      { "from": "decode1", "to": "preprocess_videomae" },
      { "from": "preprocess_videomae", "to": "action_recognition1" }
    ]
  }
}
```

### 参数说明（`i_action_recognition_node.h` 的 configure 解析）

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| model_path | string | - | TensorRT 引擎路径 |
| input_height / input_width | int | 224 | 输入分辨率 |
| num_frames / frame_interval | int | 16 / 2 | clip 帧数与采样间隔 |
| window_size / stride | int | 16 / 8 | 滑动窗口大小与步长 |
| confidence_threshold | float | 0.5 | 置信度阈值 |
| batch_size | int | 1 | 批处理大小 |
| action_labels | array | - | 类别名称列表 |

## 运行

通过 HTTP 服务提交上述管道配置：

```bash
./build/src/http/http_server 0.0.0.0 8080
curl -X POST :8080/api/v1/pipeline/build -d @config/pipelines/action_recongnition_pipeline.json
curl -X POST :8080/api/v1/pipeline/start  -d '{"id":"videomae_test"}'
```

接口详见 [api_reference.md](api_reference.md)。

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

```cpp
#include "nodes/infer/action_recognition_videomae.h"

// 方式一：默认构造 + setter
ai_stream::nodes::ActionRecognitionVideoMAENode node;
node.setModelPath("./models/action_recognition/action_recognition.engine");
node.setConfidenceThreshold(0.7f);
node.setActionLabels({"climb", "fight", "other"});
node.configure("ar1", params_json);

// 方式二：Config 结构体构造
ai_stream::nodes::ActionRecognitionVideoMAENode::Config cfg;
cfg.model_path = "./models/action_recognition/action_recognition.engine";
cfg.confidence_threshold = 0.7f;
cfg.action_labels = {"climb", "fight", "other"};
ai_stream::nodes::ActionRecognitionVideoMAENode node2(cfg);
```

> 通常无需手动集成：节点已通过 `REGISTER_NODE("action_recognition_videomae", ...)`
> 注册到工厂，在管道 JSON 中引用 type 即可。

### 结果处理

识别结果以 `InferenceResultPacket.action_result` 沿图向下传播，下游告警/融合节点按
`ActionResult::label` 与 `confidence` 处理（如 fight/climb 触发对应告警规则）。

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
5. 需以 `-DWITH_CUDA=ON -DWITH_TENSORRT=ON` 编译；GPU 帧路径要求解码输出 GPU 指针
