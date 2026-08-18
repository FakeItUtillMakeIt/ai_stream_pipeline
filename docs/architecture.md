# 架构设计说明

## 1. 总体架构

ai_stream_pipeline 是一个模块化的视频流 AI 处理框架：以 **节点图（DAG）** 组织处理逻辑，
数据以 **Packet** 形式在节点间流动，通过 HTTP API 动态构建/启停管道。

```
[HTTP Server] ---(POST JSON)---> [Pipeline / AsyncPipelineManager]
                                        |
                                        v
[Source Node] --(RawVideoPacket)--> [Decode] --(VideoFramePacket)--> [Preprocess]
                                                                          |
                                                                          v
[Sink/Encode] <--(VideoFramePacket)-- [Draw] <--(InferenceResultPacket)-- [Infer] <-- [Tracker/Alert/...]
```

核心库划分：

| 库 | 内容 |
|---|---|
| `ai_stream_core` | Node/Packet/Pipeline 核心抽象、QueuedNode、BoundedQueue、Metrics、AsyncPipelineManager |
| `ai_stream_nodes` | 全部具体节点实现 + 节点工厂注册 |
| `alert_rules` / `alert_node` | 告警规则与告警节点（对象库，并入 ai_stream_nodes） |
| `http_server` | REST API 服务可执行文件 |

## 2. 核心概念

### 2.1 Packet（数据包）

所有数据继承自 `core::BasePacket`（`include/ai_stream/core/packet.h`）：

| 类型 | 说明 |
|---|---|
| `RawVideoPacket` | 编码码流（H264/H265 NALU + extradata） |
| `VideoFramePacket` | 解码帧（CPU `cv::Mat` 和/或 GPU 指针 `d_ptr`，含 letterbox 参数） |
| `InferenceResultPacket` | 推理结果（检测框/关键点/动作识别/告警结果），携带 `source_frame` 供画框使用 |
| `STREAM_END` | 流结束信号，沿图向下传播，节点收到后自停并转发 |

公共字段：`stream_id`（多流区分）、`frame_id`、`timestamp_ms`、`cost_time_map`（各节点耗时打点）。

### 2.2 Node（节点，推模式）

`core::Node`（`include/ai_stream/core/node.h`）：

- `pushData(packet)`：上游同步调用，节点处理后通过 `broadcast()` 推给所有下游
- `configure(node_id, params)`：管道构建时由 Pipeline 调用，节点自行解析 JSON 参数（含模型加载等初始化），返回 false 则管道构建失败
- `start()/stop()`：生命周期；`broadcast()` 同时负责指标打点与失效下游清理

### 2.3 QueuedNode（背压基类）

`core::QueuedNode<Base>`（`include/ai_stream/core/queued_node.h`）是模板混入：
`class Foo : QueuedNode<IXxxNode>`。

- `pushData()` 只入队立即返回，专属 worker 线程串行执行 `processPacket()`，
  避免重处理阻塞上游
- 原 `start()/stop()` 逻辑迁移到 `onStartup()/onShutdown()` 钩子
- 满队列策略（JSON `"queue"` 字段配置）：

```json
"queue": { "capacity": 64, "drop_policy": "drop_newest", "push_timeout_ms": 10 }
```

| drop_policy | 行为 |
|---|---|
| `drop_newest`（默认） | 丢弃新包，直播场景 |
| `drop_oldest` | 丢最旧包保最新 |
| `block` | 阻塞至超时后丢弃 |

丢包计入 `MetricsCollector.dropped_packets`。STREAM_END 入队后由 worker 处理（stop + broadcast），
worker 内自停已做 join 死锁防护。

### 2.4 Pipeline（管道）

`core::Pipeline`（`include/ai_stream/core/pipeline.h` + `src/core/pipeline_manager.cpp`）：

- `buildFromJson`：工厂创建节点 → `node->configure()` → 连边 → **Kahn 算法拓扑排序 + 环检测**
- `start()`：**逆拓扑序**启动（下游先就绪，source 最后），避免数据到达未启动节点
- `stop()`：**拓扑序**停止（source 先停），阻止新数据进入，下游自然排空；启动失败时同序回滚

### 2.5 节点工厂

`REGISTER_NODE("type", ClassName)`（`src/nodes/registry/node_factory.h`）静态注册，
支持同一 .cpp 注册多个类型。Pipeline 按 JSON 中的 `type` 字符串创建实例。
告警规则同理：`REGISTER_ALERT_RULE("type", ClassName)`（`src/rules/alert/alert_rule_factory.h`）。

## 3. 节点清单与线程模型

| 节点 | type | 线程模型 |
|---|---|---|
| RTSPSourceNode | `rtsp_source` | 自持拉流线程 |
| FileSourceNode | `file_source` | 自持读文件线程（loop/realtime 可选） |
| FFmpegDecodeNode | `ffmpeg_decode` | QueuedNode |
| ResizeNormalizeNode / GPU / CUDA 版 | `resize_normalize` / `gpu_resize_normalize` / `cuda_resize_normalize` | QueuedNode |
| DetectionInferNode | `detection_infer` | 自持 BoundedQueue + 推理线程（动态 batch） |
| PoseInferNode / CudaPoseInferNode | `pose_infer` / `cuda_pose_infer` | 自持队列 + worker |
| ActionRecognitionVideoMAENode | `action_recognition_videomae` | QueuedNode |
| DetectionPostProcessNode / GPU 版 | `detection_post` / `gpu_detection_post` | QueuedNode |
| TrackerNode（OCSort/ByteTrack） | `tracker` | QueuedNode |
| AlertNode | `alert` | QueuedNode（规则可并行 std::async） |
| FusionNodeImpl | `fusion` | QueuedNode |
| OSDDrawNode / GpuOSDDrawNode | `osd_draw` / `gpu_osd_draw` | QueuedNode |
| EvidenceNode | `evidence` | 同步轻分发（内部组件各自带队列） |
| RTMPSinkNode / MP4SaveNode | `rtmp_sink` / `mp4_save` | 自持 BoundedQueue + 编码线程（drop_oldest） |

## 4. 告警规则体系

- `IAlertRule`（`include/ai_stream/rules/i_alert_rule.h`）：`initialize/process/reset/getStatistics`，
  支持多区域（`rule_zones`）与动作识别双模式（姿态/模型）
- `AlertNode` 通过 JSON `rules` 数组从工厂装配规则；`process_type: parallel|sequence`
  控制规则并行/串行执行
- 告警事件 `AlertEvent` 带状态机（occur/last/end），`toJson()` 可序列化上报
- 具体规则 20+ 种（人员入侵、安全帽、吸烟、攀爬、打架、火焰/烟雾/结晶等场景识别），
  位于 `src/rules/alert/`，复杂检测器在 `src/rules/alert/detector/`

## 5. 证据链（Evidence）

`EvidenceNode` 接收告警结果与画框帧：
- `FrameBuffer`：前 N 帧环形缓冲（pre_frames）
- `VideoRecorder`：告警触发录制 pre+post 帧证据视频（BoundedQueue，停止时排空残留帧）
- `VideoRollover`：按保留时长清理
- `FtpUploader`：证据上传（BoundedQueue 阻塞策略，证据不丢弃，带重试）

## 6. 指标与监控

`MetricsCollector` 单例（`include/ai_stream/core/metrics.h`）：
- 每节点 total/dropped/latency(min/max/avg/last)/fps 窗口统计
- 系统级 GPU 显存/CPU/管道数（由 AsyncPipelineManager 监控线程喂入）
- 输出：`/metrics`（Prometheus）、`/api/v1/metrics`（JSON）、`/api/v1/metrics`（POST 按管道查询）

## 7. HTTP 服务（双模式）

`ApiServer`（`src/http/api_server.*`）构造时选择模式（`http_server [host] [port] [--async]`）：

- **同步模式**（默认）：请求线程内直接构建/启停，响应即最终结果
- **异步模式**：委托 `AsyncPipelineManager`（任务队列 + worker 线程），
  build/start/stop 返回 `202 accepted`，生命周期状态
  `loading / load_failed / stopped / running` 通过 status 接口轮询；
  支持 `optimization.auto_batch` 按 GPU 显存自动设置推理 batch_size

详见 [api_reference.md](api_reference.md)。
