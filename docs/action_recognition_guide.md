# VideoMAE动作识别节点使用指南

## 概述

本节点实现了基于VideoMAE的视频动作识别功能，支持帧缓冲和滑动窗口机制，无需依赖pose节点。

## 功能特性

- 使用轻量化VideoMAE-Small模型（22M参数，88MB）
- 支持帧缓冲和滑动窗口机制
- 支持TensorRT加速推理
- 可配置的clip参数和滑动窗口参数
- 支持自定义动作标签

## 模型准备

### 1. 安装依赖

```bash
pip install transformers torch onnx onnxruntime
```

### 2. 导出ONNX模型

```bash
cd tools/model_converter
python videomae_export_onnx.py --output ../../models/videomae/videomae_small.onnx
```

### 3. 转换为TensorRT引擎

```bash
cd tools/model_converter
chmod +x videomae_onnx2trt.sh
./videomae_onnx2trt.sh ../../models/videomae/videomae_small.onnx ../../models/videomae/videomae_small.engine
```

## 节点参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| model_path | string | - | 模型文件路径（.engine） |
| input_height | int | 224 | 输入帧高度 |
| input_width | int | 224 | 输入帧宽度 |
| num_frames | int | 16 | 每个clip的帧数 |
| frame_interval | int | 2 | 帧采样间隔 |
| window_size | int | 16 | 滑动窗口大小 |
| stride | int | 8 | 窗口滑动步长 |
| confidence_threshold | float | 0.5 | 置信度阈值 |
| action_labels | list | - | 动作类别标签列表 |

## 管道配置示例

```json
{
  "id": "action_recognition_pipeline",
  "graph": {
    "nodes": [
      {
        "id": "source1",
        "type": "rtsp_source",
        "params": {
          "url": "rtsp://example.com/stream1"
        }
      },
      {
        "id": "decode1",
        "type": "ffmpeg_decode",
        "params": {
          "decoder_type": "h264_cuvid"
        }
      },
      {
        "id": "action_recognition1",
        "type": "action_recognition_videomae",
        "params": {
          "model_path": "models/videomae/videomae_small.engine",
          "num_frames": 16,
          "frame_interval": 2,
          "stride": 8,
          "action_labels": ["falling", "fighting", "running", "walking"]
        }
      },
      {
        "id": "draw1",
        "type": "osd_draw",
        "params": {}
      },
      {
        "id": "sink1",
        "type": "rtmp_sink",
        "params": {
          "target": "rtmp://example.com/live/stream1"
        }
      }
    ],
    "edges": [
      {"from": "source1", "to": "decode1"},
      {"from": "decode1", "to": "action_recognition1"},
      {"from": "action_recognition1", "to": "draw1"},
      {"from": "draw1", "to": "sink1"}
    ]
  }
}
```

## 工作原理

### 帧缓冲机制

节点内部维护一个帧缓冲区，按`stream_id`隔离不同视频流的帧数据。缓冲区大小由`window_size * frame_interval + 10`决定。

### 滑动窗口机制

- 每收到`stride`帧执行一次推理
- 从缓冲区中按`frame_interval`采样`num_frames`帧组成clip
- 对clip进行预处理后送入模型推理

### 推理流程

```
视频帧输入 → 帧缓冲 → 滑动窗口触发 → 采样clip → 预处理 → TensorRT推理 → 后处理 → 输出结果
```

## 性能优化建议

1. **使用FP16精度**：TensorRT转换时启用`--fp16`可大幅提升推理速度
2. **调整滑动窗口步长**：增大`stride`可降低推理频率，减少计算量
3. **调整帧采样间隔**：增大`frame_interval`可减少每帧的计算量
4. **使用GPU预处理**：可将图像预处理移到GPU上执行

## 注意事项

1. 模型输入固定为16帧、224x224分辨率
2. 支持动态batch size，但推荐使用batch_size=1
3. 动作标签顺序需与模型训练时一致
4. 置信度阈值建议设置为0.5-0.7之间
