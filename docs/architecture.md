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
| `ai_stream_hal` | 硬件抽象层：推理引擎/图像加速/编解码接口 + 各平台后端实现（静态库） |
| `ai_stream_nodes` | 全部具体节点实现 + 节点工厂注册 |
| `alert_rules` / `alert_node` | 告警规则与告警节点（对象库，并入 ai_stream_nodes） |
| `http_server` | REST API 服务可执行文件 |

依赖方向：`ai_stream_nodes → ai_stream_core / ai_stream_hal`，
节点只面向 HAL 接口编程，不直接耦合具体平台 SDK。

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
- `onIdle()` 在队列空闲（pop 超时）时回调，用于周期性任务（fusion 超时合并、
  推理节点的批次超时 flush）；派生类可用 `setPollTimeout()` / `setQueueCapacity()`
  调整空闲轮询间隔与队列容量
- 满队列策略（JSON `"queue"` 字段配置）：

```json
"queue": { "capacity": 64, "drop_policy": "drop_newest", "push_timeout_ms": 10 }
```

| drop_policy | 行为 |
|---|---|
| `drop_newest`（默认） | 丢弃新包，直播场景 |
| `drop_oldest` | 丢最旧包保最新 |
| `block` | 阻塞至超时后丢弃 |

丢包计入 `MetricsCollector.dropped_packets`。STREAM_END 是控制包：入队时**无视丢帧策略强制入队**
（必要时挤掉最旧数据），由 worker 线程调用 `processPacket()` 处理（派生类在此广播 STREAM_END），
基类随后统一 `stop()` 并调用 `onShutdown()`；worker 内自停已做 join 死锁防护，自停后可直接再次 `start()`。

### 2.4 Pipeline（管道）

`core::Pipeline`（`include/ai_stream/core/pipeline.h` + `src/core/pipeline_manager.cpp`）：

- `buildFromJson`：工厂创建节点 → `node->configure()` → 连边 → **Kahn 算法拓扑排序 + 环检测**
- `start()`：**逆拓扑序**启动（下游先就绪，source 最后），避免数据到达未启动节点；
  STREAM_END 级联自停后可直接再次 `start()`，内部自动回收残留线程并按完整拓扑重启
- `stop()`：**拓扑序**停止（source 先停），阻止新数据进入，下游自然排空；`stop()` 幂等，
  自停状态下调用仍会回收节点资源；启动失败时同序回滚

### 2.5 节点工厂

`REGISTER_NODE("type", ClassName)`（`src/nodes/registry/node_factory.h`）静态注册，
支持同一 .cpp 注册多个类型。Pipeline 按 JSON 中的 `type` 字符串创建实例。
告警规则同理：`REGISTER_ALERT_RULE("type", ClassName)`（`src/rules/alert/alert_rule_factory.h`）。

### 2.6 HAL（硬件抽象层）

`src/hal/` 将平台相关实现收敛到统一接口（接口定义在 `include/ai_stream/hal/`）：

| 接口 | 职责 | 后端 |
|---|---|---|
| `IInferenceEngine` / `IDetectionInferenceEngine` | 通用/检测推理 | TensorRT、RKNN、Ascend、地平线 BPU |
| `IActionRecognition` | 动作识别推理 | TensorRT (VideoMAE)、RKNN、Ascend |
| `IPoseEstimationEngine` | 姿态估计推理（host/设备输入/GPU 端到端三路径） | TensorRT (YOLO-Pose) |
| `IImageAccelerator` | 预处理/绘制/NMS 加速 | CUDA、NPP、RGA、CPU |
| `IVideoDecoder` | 视频解码 | FFmpeg、NVDEC、MPP、地平线 VPU(sp 硬解) |
| `IVideoEncoder` | 视频编码 | FFmpeg、MPP、地平线 VPU |

各 `*_factory.cpp` 按编译期宏（`WITH_TENSORRT` / `WITH_RKNN` / `WITH_ASCEND` / `WITH_HORIZON` /
`WITH_CUDA` / `WITH_CPU_FALLBACK`）选择可用后端；可选依赖缺失时自动降级
（如无 GPU 回退 CPU 实现）。OSD 绘制的中文渲染依赖 OpenCV freetype 模块，
缺失时通过 `HAVE_OPENCV_FREETYPE` 宏回退到 `cv::putText`。

## 3. 节点清单与线程模型

| 节点 | type | 线程模型 |
|---|---|---|
| RTSPSourceNode | `rtsp_source` | 自持拉流线程 |
| FileSourceNode | `file_source` | 自持读文件线程（loop/realtime 可选） |
| FFmpegDecodeNode | `ffmpeg_decode` | QueuedNode |
| ResizeNormalizeNode | `resize_normalize` | QueuedNode，通过 HAL 图像加速器选择 CPU/RGA/DVPP/NPP 路径 |
| DetectionInferNode | `detection_infer` | QueuedNode（worker 内批次收集，`onIdle` 超时 flush；动态 batch + CUDA Graph） |
| PoseInferNode | `pose_infer` | QueuedNode（推理走 HAL `IPoseEstimationEngine`，CPU/CUDA/RKNN 后端） |
| ActionRecognitionVideoMAENode | `action_recognition_videomae` | QueuedNode |
| DetectionPostProcessNode | `detection_post` | QueuedNode，NMS 通过 HAL 图像加速器执行 |
| TrackerNode（OCSort/ByteTrack） | `tracker` | QueuedNode |
| AlertNode | `alert` | QueuedNode（规则经常驻线程池并行执行，`process_type: parallel\|sequence`） |
| FusionNodeImpl | `fusion` | QueuedNode（双模式见下） |
| OSDDrawNode | `osd_draw` | QueuedNode（矩形框经 HAL drawBoxes 路由，GPU 数据自动走 NPP；文字/关键点/面板 CPU 绘制，中文需 OpenCV freetype，缺失时英文回退 `cv::putText`） |
| EvidenceNode | `evidence` | QueuedNode（双输入：告警触发 + 画框帧；内部 FrameBuffer/VideoRecorder/FtpUploader 各自带队列） |
| RTMPSinkNode / MP4SaveNode | `rtmp_sink` / `mp4_save` | QueuedNode（编码在 worker 线程串行执行；队列满默认丢最旧） |

> 源节点（`rtsp_source` / `file_source`）是生产者、自持读流线程，不使用 QueuedNode。
> 原生 RKNN 检测节点已并入 `detection_infer`（经 HAL `IDetectionInferenceEngine` 选择 RKNN 后端）。

> 跟踪匹配说明：TrackerNode 将检测框关联到轨迹时，优先按轨迹绑定的 `class_name`
> 匹配（轨迹诞生时按 IoU 绑定类别名），名称缺失时回退 `class_id` 比较；
> 当 `name` 与 `class_id` **同时变化**时允许类别跃迁（如 `person → fall_down`）并保持同一 track_id，
> 而多推理源融合下仅名称冲突（class_id 不冲突）则拒绝跨类继承。
> 此外 **轨迹 ID 缝合**（`stitch`）会在目标短暂丢失后，按时空邻近性把新出现的轨迹 ID
> 归并回原 ID，抑制跌倒等框形剧变场景下的 ID 跳变（详见 §5.1）。

## 4. 多推理源融合（Fusion）

FusionNode 支持两种模式（`params.mode`）：

**`action`（默认，原有行为）**：缓存动作识别结果，按时间戳阈值附加到检测包。

**`object_level`**：在 `action` 基础上，动作只附加到含匹配 `track_id` 的目标帧（`ActionResult.track_id`）；
无匹配目标的动作不附加（等待匹配帧）。动作结果被**一次性消费**，避免同一次动作重复附加到多帧。
`action_source` 可指定唯一动作来源节点（`producer_id`），非该来源的动作包原样透传、不进入缓存。

**`detection_merge`**：同一视频流喂给多个推理节点时，按 `(stream_id, frame_id)`
配对合并多路检测框：

```json
{
  "mode": "detection_merge",
  "detection_sources": ["infer1", "infer2"],
  "class_offsets": { "infer2": 14 },
  "wait_timeout_ms": 200,
  "cross_nms": false
}
```

- **来源识别**：依赖 `BasePacket.producer_id`（Node::broadcast 自动打戳为节点名，
  Pipeline 构建时节点名统一为配置 id）
- **帧配对**：全部来源到齐立即合并广播；超时（默认 200ms）广播部分合并，不阻塞后续帧
- **类别处理**：`class_offsets` 可选（默认不重映射）；class_name 始终保留。
  跟踪与告警均按 class_name 匹配，id 冲突不影响正确性；仅当下游有按 class_id
  消费的环节（draw class_filter / detection_post NMS / pose_infer）时才需配置偏移
- **跨源 NMS**：`cross_nms` 开关（默认关），同名类别框按 IoU 去重
- 兼容动作融合：合并结果仍会附加时间戳匹配的动作识别缓存
- STREAM_END：排空所有待合并帧后自停并转发

## 5. 告警规则体系

- `IAlertRule`（`include/ai_stream/rules/i_alert_rule.h`）：`initialize/process/reset/getStatistics`，
  支持多区域（`rule_zones`）与动作识别双模式（姿态/模型）
- `AlertNode` 通过 JSON `rules` 数组从工厂装配规则；`process_type: parallel|sequence`
  控制规则并行/串行执行
- 告警事件 `AlertEvent` 带状态机（occur/last/end），`toJson()` 可序列化上报
- 具体规则 20+ 种（人员入侵、安全帽、吸烟、攀爬、打架、火焰/烟雾/结晶等场景识别），
  位于 `src/rules/alert/`，复杂检测器在 `src/rules/alert/detector/`
- 规则统一继承 `AlertRuleBase`（`src/rules/alert/alert_rule_base.h`）：基类实现 `process()`
  模板（加锁 → `onPreProcess()` 每帧钩子 → 逐 zone 调 `rule_logic()` → 事件聚合/状态机衰减）
  以及 `parseZones()` / `updateZoneEvent()`；派生类只需实现 `rule_logic()`（特征判定）与
  `initialize()`；有状态检测器在 `onPreProcess()` 中**每帧仅运行一次**，避免多区域下重复推进状态机

### 5.1 跌倒相关规则（`fall_down` vs `falling`）

| 注册键 | 规则 | 判定方式 |
|---|---|---|
| `fall_down` | `FallDownRule` | **静态**：当前帧存在 `fall_down` 类别框并持续 `alert_duration_ms` |
| `falling` | `FallingRule` | **过程**：基于 track_id 追踪 `person → down` 类别转移，且 down 状态持续 `down_confirm_ms` |

`falling` 依赖 TrackerNode 在类别跃迁时保持同一 track_id（见 §3）以及 ID 缝合（抑制跌倒时
tracker 重建轨迹导致 ID 跳变）。配置示例：

```json
{
  "type": "falling",
  "params": {
    "name": "falling1",
    "person_class": "person",
    "down_class": ["fall_down"],
    "down_confirm_ms": 1000,
    "track_timeout_ms": 5000,
    "rule_zones": []
  }
}
```

- `person_class` 默认 `"person"`，`down_class` 默认 `["down"]`；**需按检测模型实际类别名配置**
  （如本仓库 14 类模型输出 `fall_down`，须写 `"down_class": ["fall_down"]`）
- `down_class` 支持字符串或字符串数组；`down_confirm_ms` 为 down 状态需持续的毫秒数
- 未观测到 `person` 直接出现的 down 不触发；down 过程中恢复为 person 则取消本次判定
- 告警类型复用 `AlertType::FALL_DOWN`，`object_ids` 携带跌倒目标 track_id

## 6. 证据链（Evidence）

`EvidenceNode` 接收告警结果与画框帧：
- `FrameBuffer`：前 N 帧环形缓冲（pre_frames）
- `VideoRecorder`：告警触发录制 pre+post 帧证据视频（BoundedQueue，停止时排空残留帧）
- `VideoRollover`：按保留时长清理
- `FtpUploader`：证据上传（BoundedQueue 阻塞策略，证据不丢弃，带重试）

## 7. 指标与监控

`MetricsCollector` 单例（`include/ai_stream/core/metrics.h`）：
- 每节点 total/dropped/latency(min/max/avg/last)/fps 窗口统计
- 系统级 GPU 显存/CPU/管道数（由 AsyncPipelineManager 监控线程喂入）
- 输出：`/metrics`（Prometheus）、`/api/v1/metrics`（JSON）、`/api/v1/metrics`（POST 按管道查询）

## 8. HTTP 服务（双模式）

`ApiServer`（`src/http/api_server.*`）构造时选择模式（`http_server [host] [port] [--async]`）：

- **同步模式**（默认）：请求线程内直接构建/启停，响应即最终结果
- **异步模式**：委托 `AsyncPipelineManager`（任务队列 + worker 线程），
  build/start/stop 返回 `202 accepted`，生命周期状态
  `loading / load_failed / stopped / running` 通过 status 接口轮询；
  支持 `optimization.auto_batch` 按 GPU 显存自动设置推理 batch_size

详见 [api_reference.md](api_reference.md)。
