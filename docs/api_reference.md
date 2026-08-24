# HTTP API 参考

服务启动：`http_server [host] [port] [--async]`（默认 `0.0.0.0:8080` 同步模式）。

管道配置格式（build 请求体）：

```json
{
  "id": "my_pipeline",
  "graph": {
    "nodes": [
      { "id": "src1", "type": "rtsp_source", "params": { "url": "rtsp://..." } },
      { "id": "decode1", "type": "ffmpeg_decode", "params": { "output_bgr": true } }
    ],
    "edges": [
      { "from": "src1", "to": "decode1" }
    ]
  }
}
```

节点 `params` 由各节点 `configure()` 自行解析；QueuedNode 节点额外支持可选队列配置：

```json
"queue": { "capacity": 64, "drop_policy": "drop_newest|drop_oldest|block", "push_timeout_ms": 10 }
```

### 可用节点类型

| type | 节点 | 说明 |
|---|---|---|
| `rtsp_source` | RTSPSourceNode | RTSP 拉流 |
| `file_source` | FileSourceNode | 文件源（loop/realtime 可选） |
| `ffmpeg_decode` | FFmpegDecodeNode | FFmpeg 解码 |
| `resize_normalize` / `gpu_resize_normalize` / `cuda_resize_normalize` | 预处理 | CPU/GPU/CUDA 三版 |
| `detection_infer` | DetectionInferNode | TensorRT 目标检测（动态 batch） |
| `rknn_detection_infer` | RknnDetectionInferNode | RK3588 目标检测 |
| `pose_infer` / `cuda_pose_infer` | 姿态估计 | CPU/CUDA 版 |
| `action_recognition_videomae` | ActionRecognitionVideoMAENode | VideoMAE 动作识别 |
| `detection_post` / `gpu_detection_post` | 后处理（解码/NMS） | CPU/GPU 版 |
| `tracker` | TrackerNode | OCSort/ByteTrack 跟踪 |
| `alert` | AlertNode | 告警规则容器 |
| `fusion` | FusionNodeImpl | 多推理源融合 |
| `osd_draw` / `gpu_osd_draw` | OSD 绘制 | 中文绘制需 OpenCV freetype |
| `evidence` | EvidenceNode | 证据链录制/上传 |
| `rtmp_sink` / `mp4_save` | 推流/存盘 | 编码输出 |

> 具体节点参数请参考各节点的 `configure()` 实现及
> `config/pipelines/` 下的示例管道 JSON。

---

## POST /api/v1/pipeline/build — 构建管道

请求体：如上。

**同步模式**：构建完成才返回（含模型加载，可能耗时较长）。

```json
200 {"status": "ok", "id": "my_pipeline", "message": "Pipeline built successfully"}
400 {"error": "Invalid pipeline configuration"}   // 配置错误/构建失败
409 {"error": "Pipeline id already exists"}
```

**异步模式**：提交构建任务立即返回，状态轮询 `/api/v1/pipeline/status`。

```json
202 {"status": "accepted", "id": "my_pipeline", "state": "loading",
     "message": "Pipeline build submitted, poll /api/v1/pipeline/status for progress"}
```

## POST /api/v1/pipeline/start — 启动管道

请求体：`{"id": "my_pipeline"}`

- 同步：`200 {"status":"ok","id":"...","running":true}`；启动失败 `500`
- 异步：`202 {"status":"accepted","id":"...","state":"starting"}`
- 管道不存在：`404 {"error":"Pipeline not found"}`

## POST /api/v1/pipeline/stop — 停止管道

请求体：`{"id": "my_pipeline"}`

- 同步：`200 {"status":"ok","id":"...","running":false}`
- 异步：`202 {"status":"accepted","id":"...","state":"stopping"}`
- 不存在：`404`

## DELETE /api/v1/pipeline/delete — 销毁管道

请求体：`{"id": "my_pipeline"}`（先停止再从管理器移除）

- 同步：`200 {"status":"ok","id":"...","message":"Pipeline destroyed"}`
- 异步：`202 {"status":"accepted","id":"...","message":"Pipeline destroy submitted"}`
- 不存在：`404`

## GET /api/v1/pipeline/list — 管道列表

```json
// 同步模式
{"pipelines": [{"id": "my_pipeline", "running": true}]}

// 异步模式（多 state 字段）
{"pipelines": [{"id": "my_pipeline", "state": "running", "running": true}]}
```

## POST /api/v1/pipeline/status — 管道状态

请求体：`{"id": "my_pipeline"}`

```json
// 同步模式
{"id": "my_pipeline", "running": true}

// 异步模式：state ∈ loading | load_failed | stopped | running
{"id": "my_pipeline", "state": "running", "running": true}
```

不存在：`404 {"status":"error","message":"Pipeline not found","id":"..."}`

> 注：
> - 节点收到 STREAM_END（如文件播放完毕）会级联自停，此时 `running` 以节点实际状态为准
> - 自停后的管道可直接再次调用 start 重启，无需先 stop

## GET /health — 健康检查

```json
{"status": "healthy", "mode": "sync|async", "timestamp": 1787023825}
```

## GET /metrics — Prometheus 指标

返回 `text/plain` Prometheus 格式：各节点处理量/丢包/延迟/FPS + 系统 GPU 显存/CPU/管道数。

## GET /api/v1/metrics — 全量 JSON 指标

```json
{
  "pipelines": {
    "my_pipeline": [
      {"node_name": "DetectionInfer", "total_packets": 2624, "dropped_packets": 0,
       "min_latency_ms": 0, "max_latency_ms": 0, "fps": 310.7}
    ]
  },
  "system": {"gpu_memory_mb": 0, "cpu_percent": 0, "active_pipelines": 1, "total_pipelines": 1}
}
```

## POST /api/v1/metrics — 单管道节点指标

请求体：`{"id": "my_pipeline"}`

```json
{
  "pipeline_id": "my_pipeline",
  "nodes": [
    {"node_name": "DetectionInfer", "total_packets": 2624, "dropped_packets": 0,
     "latency_ms": {"avg": 3, "min": 2, "max": 9, "last": 3}, "fps": 310.7}
  ]
}
```

无指标：`404 {"error": "No metrics found for pipeline"}`

---

## 异步模式典型调用序列

```bash
# 1. 提交构建
curl -X POST :8080/api/v1/pipeline/build -d @pipeline.json
# 2. 轮询直到 state 变为 stopped（构建完成）或 load_failed
curl -X POST :8080/api/v1/pipeline/status -d '{"id":"my_pipeline"}'
# 3. 启动并轮询 running
curl -X POST :8080/api/v1/pipeline/start -d '{"id":"my_pipeline"}'
# 4. 停止 / 销毁
curl -X POST :8080/api/v1/pipeline/stop -d '{"id":"my_pipeline"}'
curl -X DELETE :8080/api/v1/pipeline/delete -d '{"id":"my_pipeline"}'
```
