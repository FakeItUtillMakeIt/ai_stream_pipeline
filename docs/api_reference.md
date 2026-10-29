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

### 鉴权与角色（研发 / 实施）

- **可选鉴权**：`AI_STREAM_API_TOKEN`=研发/管理员 token；`AI_STREAM_IMPL_TOKEN`=实施 token。设置任一即启用；所有 `/api/` 接口需带 `Authorization: Bearer <token>`，否则 `401`。都不设置则视为开放（角色=dev，向后兼容）。
- **角色判定**：`GET /api/v1/whoami` → `{role: dev|impl|anon, auth_enabled}`。匹配 admin token→`dev`，匹配 impl token→`impl`。
- **权限矩阵**：
  - `impl` 允许：`pipeline/build|start|stop|delete|list|status|get-config`、`alerts/*`、`evidence/*`、`selfcheck`、`templates`(list/get)。即实施可实例化/启停/查历史，但**不能改模板与保存配置**。
  - **dev-only（impl 调用返回 403）**：`pipeline/save-config`、`templates/save`、`templates/delete`。
- 前端：`whoami` 返回非 dev 时，界面锁在实施台（进不去 `#/editor`，隐藏"发布为模板/保存到服务器"）；后端 403 为最终防线。
- 请求体上限 **16 MB**，读超时 10s，写超时 60s。
- 错误码约定：

| 状态码 | 含义 |
|---|---|
| 400 | 参数缺失 / JSON 解析失败 / 配置非法 |
| 401 | 未通过 token 鉴权 |
| 403 | 角色权限不足（impl 调 dev-only 接口） |
| 404 | 管道不存在 |
| 409 | 管道 id 已存在 |
| 413 | 请求体超过上限 |
| 500 | 内部错误 |

- 服务默认监听 `0.0.0.0`，生产环境建议绑定内网/回环地址并开启 token。

### 可用节点类型

| type | 节点 | 说明 |
|---|---|---|
| `rtsp_source` | RTSPSourceNode | RTSP 拉流 |
| `file_source` | FileSourceNode | 文件源（loop/realtime 可选） |
| `ffmpeg_decode` | FFmpegDecodeNode | FFmpeg 解码 |
| `resize_normalize` | 预处理 | 通过 HAL 图像加速器选择 CPU/RGA/DVPP/NPP 路径 |
| `detection_infer` | DetectionInferNode | 目标检测（HAL：TensorRT/RKNN/BPU；动态 batch） |
| `pose_infer` | 姿态估计 | 推理走 HAL `IPoseEstimationEngine`（CPU/CUDA/RKNN） |
| `action_recognition_videomae` | ActionRecognitionVideoMAENode | VideoMAE 动作识别 |
| `detection_post` | 后处理（NMS） | 通过 HAL 图像加速器执行 |
| `tracker` | TrackerNode | OCSort/ByteTrack 跟踪（类别跃迁 + ID 缝合） |
| `alert` | AlertNode | 告警规则容器 |
| `fusion` | FusionNodeImpl | 多推理源融合 |
| `osd_draw` | OSD 绘制（CPU/GPU 自适应，框绘制走 HAL） | 中文绘制需 OpenCV freetype |
| `evidence` | EvidenceNode | 证据链录制/上传；落告警快照后转发事件给下游（见下） |
| `relation_recognition` | RelationRecognitionNode | 视觉关系三元组检测（YOLOE+RelateAnything，见 `relation_chain.md`） |
| `vlm_gate` | VlmGateNode | VLM 告警核验闸门：判真才放行（需 `WITH_VLM`，见 `vlm_alert_gate.md`） |
| `report` | ReportSinkNode | 告警上报 webhook，附带 base64 报警图（需 `WITH_VLM`） |
| `rtmp_sink` / `mp4_save` | 推流/存盘 | 编码输出 |

> 具体节点参数请参考各节点的 `configure()` 实现及
> `config/pipelines/` 下的示例管道 JSON。

`tracker` 可选配置（轨迹 ID 缝合，抑制跌倒等框形剧变导致的 ID 跳变）：

```json
{
  "tracker_type": "ocsort",
  "sub_stream_id": "src1",
  "ocsort_config": { "iou_threshold": 0.3, "inertia": 0.2, "max_age": 30 },
  "stitch": { "enabled": true, "gap_frames": 15, "dist_ratio": 0.5, "memory_frames": 300 }
}
```

> `falling` 告警规则的 `down_class` 必须与检测模型实际类别名一致
> （本仓库 14 类模型输出 `fall_down`，须写 `"down_class": ["fall_down"]`）。

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
{"pipelines": [{"id": "my_pipeline", "running": true, "play_url": "...", "evidence_dirs": ["evidence"]}]}

// 异步模式（多 state 字段）
{"pipelines": [{"id": "my_pipeline", "state": "running", "running": true, "play_url": "...", "evidence_dirs": ["evidence"]}]}
```

`play_url` / `evidence_dirs` 由 build 时从管道图抽取（预览页使用），详见"证据与预览"：
- `play_url`：输出侧 sink 节点（`rtmp_sink`/`mp4_save`/`hls_sink`）`params.play_url`，即媒体服务器转出的 HLS(.m3u8)/WebRTC 播放地址；未配置为空串。
- `evidence_dirs`：`evidence` 节点 `video.output_dir` 与 `snapshot.output_dir` 之和（相对程序运行目录）。

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

## GET /api/v1/pipeline/get-config?id=<id>
返回该管道**已保存的配置**（`build`/`save-config` 时落盘的 `{id, graph}`），供编辑器"从服务器同步/导入"外部（如 Apipost/平台）创建的管道。
- 无配置：`404 {"error":"config not found","id":"..."}`。
> 编辑器"同步服务器"按钮 + 每 6s 静默轮询：拉 `list` 刷新运行态，并把"服务器有、本地无"的管道用本接口导入。预览页每 4s 轮询 `list`。

## GET /api/v1/selfcheck?id=<id>
实施页环境自检：后端真实探测该管道健康度，返回结构化 `checks`（每项 `key/label/level/detail`，`level ∈ ok|warn|err|idle`）。
探测项：
- `state` 运行状态（running/stopped/load_failed/…）
- `source` 拉流地址：`file://` 查文件是否存在；`rtsp/rtmp` 做 TCP 连通探测
- `model` 引擎/模型：递归扫描节点参数里的 `.engine/.onnx/.pth/.npz` 路径，按服务器 CWD 解析后查存在
- `push` RTMP 推流：解析 `rtmp_sink.output_url` 的 host:port 做 TCP 探测（mediamtx 是否可达）
- `play` 播放地址：对 `play_url` 发 HTTP(HEAD)，200→可播、404→无 publisher、连不上→err（CORS 属浏览器侧，服务端探不出）
- `evidence` 证据目录：解析绝对路径 + 统计文件数（不存在→err，空→warn）
```json
{"id":"p1","checks":[{"key":"play","label":"播放地址","level":"warn","detail":"404 无 publisher（流没推上/路径不对）：http://…"}]}
```
> 探测含网络超时（TCP 1.5s、HTTP 3s），故编辑器仅在载入/手动"服务器同步"/"重新自检"时调用，6s 自动轮询只刷新状态不调用本接口。

## 模板系统（研发发布 / 实施选用）
模板存于 `<AI_STREAM_STATE_DIR>/templates/<name>.json`，结构：
```json
{ "name":"周界入侵", "version":"3", "description":"…",
  "doc": { "id":"…", "graph": { "nodes":[…], "edges":[…] } },
  "overrides": {
    "source_url": { "node":"src1", "path":"params.url", "label":"拉流地址" },
    "push":       { "node":"rtmp_sink", "label":"是否推流" },
    "alerts":     { "node":"alert", "types":["person_intrusion","fighting"], "label":"告警项" }
  } }
```
`overrides` 声明"暴露给实施"的字段，其余参数锁定；实施台从模板实例化时只按 overrides 渲染控件并注入值。

- `GET /api/v1/templates` → `{"templates":[{name,version,description}]}`
- `GET /api/v1/templates/get?name=<name>` → 完整模板 JSON；不存在 `404`
- `POST /api/v1/templates/save` → 发布/更新（body 为模板对象，需 `name` + `doc.graph`）；名含 `/ \ ..` 或空 → `400`
- `POST /api/v1/templates/delete` → body `{name}`；不存在 `404`

> 前端：研发编辑器「发布为模板」（自动探测 源/rtmp_sink/alert 节点生成 overrides）；实施台「新增一路」从模板库选择并实例化。

## 告警历史（记录 / 检索 / 确认）
**分层记录**，按 `uid = pipeline#alert_id` merge 归并（同一告警一行，前后对比）：
- **闸门之前**（`evidence` 节点，`pre_gate`）：所有触发的告警都记（不受 VLM 闸门判决影响），含 `triggered/alert_name/snapshot`。
- **闸门之后**（`vlm_gate` 节点，`post_gate`）：判决 `decision(PUSH|HOLD|DROP|PASSTHROUGH)/verdict/confidence/reason/error/pushed/latency_ms/model/snapshot`。
- `report` 节点在告警 PUSH 到平台时也会补 `snapshot/vlm`（可选，无 gate 时兜底）。
存储：`<state>/alerts-YYYYMMDD[.N].jsonl`（事件，**按天轮转**，同天超 `AI_STREAM_ALERT_MAX_FILE_MB` 再切片）+ `<state>/alert_proc.json`（处理状态）。启动只加载最近 `AI_STREAM_ALERT_LOAD_DAYS` 天、内存裁 `cap`(5000) 条；删除超过 `AI_STREAM_ALERT_RETENTION_DAYS` 天的旧文件。记录字段：`uid/pipeline/ts/type/level/description/proc/pre_gate/post_gate/snapshot/vlm/alert`。
> 环境变量：`AI_STREAM_ALERT_RETENTION_DAYS`(默认90)、`AI_STREAM_ALERT_MAX_FILE_MB`(默认50)、`AI_STREAM_ALERT_LOAD_DAYS`(默认90)。

- `GET /api/v1/alerts?pipeline=&type=&proc=&from=&to=&q=&limit=&offset=` → `{total, items[]}`（新→旧）
  - `proc ∈ all|new|ack|dismiss|resolve`；`from/to` epoch 毫秒；`q` 匹配 描述/类型/管道。
- `POST /api/v1/alerts/ack` → `{uid, proc, note?, by?}`；非法 proc `400`，不存在 `404`。
- `GET /api/v1/alerts/stats?pipeline=` → `{new,ack,dismiss,resolve,total}`
- `GET /api/v1/alerts/file?path=` → 取告警快照（限服务器运行目录内，越界 `403`；Range 由 httplib 处理）

> 前端 `#/alerts`：时间线 + 过滤(管道/类型/处理状态/**闸门判决**/时间/搜索) + 每行显示「① 触发(证据) / ② 闸门(PUSH/HOLD/DROP+conf)」前后对比 + 缩略图 + 确认/误报/已解决 + 详情 JSON。
> 注：需 `evidence` 节点记录闸门之前；有 `vlm_gate` 才有 `post_gate`。这样即使闸门 HOLD/DROP（report 收不到）也能在历史里看到并对比。

## 证据与预览（预览页专用）

实时预览（方案 A）：浏览器不能直接播 RTMP。`rtmp_sink` 推到外部媒体服务器（mediamtx / nginx-rtmp），由其转成 **HLS(.m3u8) 或 WebRTC**；把该播放地址填进 sink 节点的 `play_url`，前端预览页用 hls.js 播放。后端不转码，仅提供播放地址来源与证据文件。

### GET /api/v1/evidence/list?pipeline=<id>
列出该管道 `evidence_dirs` 内的证据文件（快照图 / 录像片段）：
```json
{"pipeline": "my_pipeline", "play_url": "...", "files": [
  {"dir_index": 0, "name": "alert_1.jpg", "kind": "image", "size": 88123},
  {"dir_index": 0, "name": "fight_2.mp4", "kind": "video", "size": 456789}
]}
```
目录不存在或管道无证据配置 → `files` 为空数组。`kind ∈ image | video | other`。

### GET /api/v1/evidence/file?pipeline=<id>&dir=<idx>&name=<filename>
取回单个证据文件的字节（`dir` 是 `evidence/list` 返回的 `dir_index`）。
- 安全：`name` 不得含 `/ \ ..`，且解析后必须仍位于该 `dir_index` 对应目录内，否则 `400/403`；文件不存在 `404`。
- `Content-Type` 按扩展名给出（image/jpeg、image/png、video/mp4、video/webm）。
- 支持 **HTTP Range**（`Accept-Ranges: bytes` + `206 Partial Content`），`<video>` 播放/拖动定位正常。

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

---

## 请求持久化与掉线/重启自动恢复

程序掉线或板子重启后，服务会依据历史请求自动把管道恢复到崩溃前的状态，无需平台重发。

### 状态目录

- 默认 `./api_state/`，可用环境变量 `AI_STREAM_STATE_DIR` 覆盖。
- 结构：
  - `api_state/requests.log` —— 历史文本，**每行一条 JSON 记录**，最新在最后一行：
    ```
    {"ts":1791776804425,"action":"build","id":"pipeB","config":"api_state/pipelines/pipeB.json"}
    ```
    `action` ∈ `build | start | stop | delete`。
  - `api_state/pipelines/<id>.json` —— build 请求体（含 `id` + `graph`），即"配置文件"，恢复时据此重建。

### 记录时机

每个接口**成功**后追加一行：`build` 会把请求体存成配置文件；`start/stop/delete` 复用同一配置文件路径。同步/异步两种模式都会记录。

### 恢复规则（按每个 pipeline 的最后状态独立恢复）

服务启动、就绪后读取 `requests.log`，对**每个 id 取其最后一条记录**决定：

| 末态 action | 恢复动作 |
|---|---|
| `build` | 重建管道（**不启动**）——build 过就恢复其"已构建待命"状态 |
| `start` | 重建 + 启动 |
| `stop` | 重建（不启动） |
| `delete` | 跳过（不复活） |
| 配置文件缺失/坏行 | 跳过并告警 |

即：**只有处于 `start` 末态的管道会被重新拉起运行**；`build`/`stop` 末态只重建到"待命"；`delete` 不恢复。多管道各自独立恢复（不是只看全局最后一行）。

### 恢复日志

```
[Recover] 'pipeA' 末态=build → 重建（不启动）
[Recover] 'pipeB' 末态=start → 重建+启动
[Recover] 'pipeC' 末态=stop  → 重建（不启动）
[Recover] 'pipeD' 末态 delete，跳过
[Recover] 完成：重建 3（含启动 1），跳过 1
```

### 注意

- 恢复是"尽力而为"：重建会重新加载模型，若资源不可用会记 `重建失败/启动失败` 并继续处理其余管道。
- 想清空恢复状态：删除 `api_state/` 即可（下次起服务视为无历史，等待平台下发）。
- `delete` 目前只记历史、不删配置文件；如需彻底清理，手动删 `api_state/pipelines/<id>.json`。
