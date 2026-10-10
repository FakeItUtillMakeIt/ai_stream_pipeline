# VLM 告警闸门与上报（vlm_gate / report）

在关系/规则产生告警后，接入一个多模态大模型（VLM）做**推送前的硬闸门**：
`只有 VLM 判真（且置信度达标）才对外上报`，判假丢弃，异常/存疑转人工，不丢弃。
同时 `report` 节点上报时**把报警图以 base64 附带上传**。

实现全部落在 `ai_stream_pipeline` 内作为**节点**，复用已有 libcurl 依赖，不引入新服务进程。

---

## 1. 接线

```
实时预览:   rel1 → alert1 → draw1 → sink1           (画面照常，不受影响)
告警支路:   alert1 → evidence1 → vlm_gate → report   (核验后只推判真)
未接 gate:  alert1 → evidence1 → report              (按告警直接上报)
```

- **"是否走 VLM"靠边切换，节点内不做 if**：`report` 只上报"它收到的"事件。
  上游接的是 `evidence`（无 gate）还是 `vlm_gate`（有 gate），决定上报范围。
- `vlm_gate.enabled=false` 时直接透传（等价没接 gate），是灰度回退开关。
- 仓库示例 `child_near_boundary_pipeline.json` 里 `vlm_gate` 与 `report` **默认
  `enabled=false`**，行为与接入 VLM 前完全一致（不发任何外部请求）。启用见 §7。

---

## 2. evidence 的改动：关联标注图并转发

`evidence` 原本只消费告警、不转发。现在在**告警支路**上多下一站：

```cpp
// evidence_node.cpp trySavePendingSnapshot（落快照后）
const std::string path = writeSnapshot(frame, *event);   // frame 来自 draw→evidence，已画框
// 把这张已标注快照的路径回挂到"携带告警的那个包"，再广播下游（gate/report）
for (result : packet->alert_result)
    for (ev : result.alert_events)
        if (ev.status == OCCUR && !ev.extra_data.contains("snapshot_path"))
            ev.extra_data["snapshot_path"] = path;
broadcast(packet);
```

要点：
- 送 VLM / 上报的图是 **evidence 存的、已画框的标注图**，不是原始帧。
  这样"VLM 看到的 = 存证看到的 = 上报附带的"，三者一致。
- 关联键用现成的 `AlertEvent`（`alert_id`/`zone_no`/`object_ids`）。
- `broadcast` 在无下游时是空操作 → 对既有管道**向后兼容**。
- 注意：evidence 仍**不透传实时帧流**（`DECODED_FRAME`），所以 `evidence` 依旧不能
  串在 `draw → sink` 之间；它只在"落了一张告警快照"时转发那一个 `META_DATA` 包。

---

## 3. vlm_gate 决策语义

收到带 `extra_data.snapshot_path` 的告警事件后：

```
读 snapshot_path → 读图(可 max_side 下采样) → base64 → POST <url>/chat/completions
解析 content 里的 {"verdict","confidence","reason"}：

 verdict=true  且 confidence ≥ 该类阈值           → PUSH   （放行给 report）
 verdict=false                                       → DROP   （不上报，落审计）
 verdict=true  但 confidence < 阈值                    → HOLD   （不上报，转人工）
 无图 / 图不可读 / 超时 / 非2xx / JSON 不可解析         → HOLD   （不上报，不丢弃，转人工）
```

- **`error` ≠ 判假**：VLM 挂了不产生 `false`，走 HOLD，既不误报也不"因宕机丢弃真告警"，
  而是落人工复核队列。
- **只有 PUSH（或未配模板且 `unmatched_policy=passthrough` 的 PASSTHROUGH）事件会被
  broadcast**；全 DROP/HOLD 时 gate 不发任何东西（report 收不到即不上报）。
- **每类独立阈值**：`prompts[名字].confidence_threshold` 覆盖全局 `confidence_threshold`，
  不同告警（把握度不同）可分别调。
- **去重/冷却**：`dedupKey = stream_id | alert_type | zone_no | 首个 object_id`；
  命中 `cooldown_s` 内不重复调 VLM、不重复放行。多类告警天然按类型隔离，互不压制。
- **审计**：每个决策写一行 `review_dir/decisions.jsonl`，字段含
  `decision / vlm_verdict / vlm_confidence / vlm_reason / vlm_error / latency_ms /
  snapshot_path / object_ids / ground_truth(null)`，供漏报误报评测与后续微调。

### 多告警类型：一套 gate 处理所有（P0）

gate 与告警类型解耦——决策/去重/审计/附图逻辑通用，**只有"核验问题"按类型不同**：

```
解析优先级：prompts[alert_name] 命中 → 用其精调问题
            未命中 → 按 unmatched_policy：
              auto（默认）: 用 question_template 把 {alert} 换成 prettify(alert_name) 自动生成
              passthrough : 不调 VLM，直接放行（不阻塞未覆盖的告警）
              hold        : 不调 VLM，转人工
```

即：任何告警挂上 gate 都能核验（auto 按名字生成问题）；想更准就给它加一条
`prompts`。示例：

```jsonc
"vlm_gate": {
  "url":"http://ip:8901/v1","model":"qwen3.8-27b","api_key":"${VLM_API_KEY}",
  "enable_thinking": false, "max_tokens": 512, "confidence_threshold": 0.6,
  "unmatched_policy": "auto",
  "prompts": {
    "child_near_wall": { "question":"儿童是否在墙/门附近，有翻越坠落风险？", "confidence_threshold":0.6 },
    "fall_down":       { "question":"画面中是否有人跌倒/倒地不起？", "confidence_threshold":0.7 },
    "fighting":        { "question":"是否发生斗殴肢体冲突？" }
  }
}
```

> 注意：跌倒/打架这类**时序动作**用单张快照把握差，容易误判 false；这类建议走 P1
> 的多帧输入（`frames>1`），P0 只保证"能按名字核验"。

### 本地/云端统一

不区分后端：只要一个 OpenAI 兼容端点 `url + api_key + model`。本地 vLLM/SGLang
起 OpenAI 兼容服务、或云端 `qwen-vl` 兼容端点，配置格式完全一致，切换只改值。

`api_key` 支持 `${NAME}` 环境变量展开（默认明文，切 env 只改值不改代码）。

---

## 4. report 上报：payload 与 base64 图

`report` 对**每个收到的事件** POST 一次 webhook。payload：

```jsonc
{
  "source": "ai_stream",
  "report_ts_ms": 1791514937620,
  "stream_id": 1001,
  "timestamp_ms": 5708,
  "alert": {                 // = AlertEvent::toJson()
    "alert_name": "child_near_wall",
    "alert_type_name": "child_near_boundary",
    "status": 0, "status_name": "occur",
    "zone_no": 0, "object_ids": [2],
    "detect_ms": 4791, "duration_ms": 917,
    "extra_data": {
      "snapshot_path": "./evidence/snapshots/child_near_wall_....jpg",
      "vlm": { "decision":"PUSH","verdict":true,"confidence":0.9,"reason":"儿童在墙边" }
    }
  },
  "snapshot_base64": "<JPEG base64>",   // 字段名 = image_field，attach_image=true 时存在
  "image_mime": "image/jpeg"
}
```

- 图片取自 `alert.extra_data["snapshot_path"]`（evidence 的标注图），base64 注入
  `image_field`（默认 `snapshot_base64`）。
- `image_max_side>0` 先下采样重编码（JPEG q=85）控体积；`0` 则原样读文件字节编码（不重编码）。
- **缺图/读失败降级为"无图上报"**，绝不因图片问题丢掉告警本身。
- 重试 `retries` 次仍失败只告警日志（不阻塞 worker），失败也计入 `[Report] pushed ... FAILED`。

---

## 5. 节点配置

`vlm_gate`（`VlmGateConfig`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | false 透传（未接入 VLM） |
| `url` | 必填 | OpenAI 兼容基址，如 `http://ip:8901/v1` |
| `api_key` | "" | 支持 `${ENV}`；可留空（本地无鉴权） |
| `model` | 必填 | 模型名 |
| `endpoint` | `/chat/completions` | 追加到 url |
| `timeout_ms` / `connect_timeout_ms` | 3000 / 1500 | curl 超时；超时→HOLD。真实模型（尤其思考/本地7B）建议 8000~15000 |
| `retries` | 1 | VLM 失败重试 |
| `max_side` | 1024 | 送图前下采样 |
| `max_tokens` | 512 | 生成上限。**思考模型必须够大**，否则 content 被 reasoning 吃光→空→HOLD |
| `enable_thinking` | false | 关思考（注入 `chat_template_kwargs.enable_thinking=false`）：更快、content 直接是 JSON |
| `confidence_threshold` (τ) | 0.6 | 全局阈值：true 且 conf≥τ 才 PUSH；可被 `prompts[].confidence_threshold` 覆盖 |
| `cooldown_s` | 30 | 同目标冷却 |
| `review_dir` | `./vlm_review` | 审计 JSONL |
| `passthrough` | false | true 直接全放行（灰度回退） |
| `system_prompt` | 见实现 | 全局 system（只输出 JSON） |
| `question_template` | 见实现 | `unmatched_policy=auto` 时生成问题用，`{alert}` 占位 alert_name |
| `prompts` | {} | `alert_name → {question, system?, confidence_threshold?, max_tokens?}` 精调表 |
| `unmatched_policy` | `auto` | 未配 prompts 的类型：`auto`/`passthrough`/`hold` |

> **换真实模型"没反应"的头号原因**：思考模型 + `max_tokens` 太小 → `content` 为空 →
> 解析失败 → 每帧 HOLD、不外发。设 `enable_thinking=false` 且 `max_tokens≥512` 即解决
> （实测该配置下延迟从"空返回"降到 ~1.5s 且判真）。

`report`（`ReportConfig`）：

| 参数 | 默认 | 说明 |
|---|---|---|
| `enabled` | true | false 完全不上报 |
| `url` | 必填 | webhook 全地址 |
| `api_key` | "" | 支持 `${ENV}`，转 `Authorization: Bearer` |
| `headers` | [] | 额外请求头字符串数组 |
| `timeout_ms`/`connect_timeout_ms`/`retries` | 3000/1500/1 | 上报调用 |
| `source` | `ai_stream` | payload 来源标识 |
| `attach_image` | true | 是否附带 base64 图 |
| `image_max_side` | 0 | >0 下采样；0 原样 |
| `image_field` | `snapshot_base64` | 图片字段名 |

---

## 6. 构建

`CMakeLists.txt` 新增 `option(WITH_VLM ON)`。gate/report 依赖 libcurl：

- `WITH_FTP=ON` 时已有 `CURL::libcurl`；
- `WITH_FTP=OFF` 时单独 `find_package(CURL)`；找不到自动 `WITH_VLM=OFF`（不阻断整包）。

管道里若引用 `vlm_gate`/`report` 但 `WITH_VLM=OFF`（curl 缺失），工厂找不到该类型 →
构建节点失败、管道起不来。用这两个节点务必保证 curl 可用、`WITH_VLM=ON`。

---

## 7. 启用步骤（本地）

1. 起 VLM 兼容端点（或先用 mock，见 §8）。
2. 配置里把 `vlm1.params.enabled=true` 并填 `url/model/api_key`；`report1.params.enabled=true` 填 `url`。
   真实思考模型务必 `enable_thinking:false` + `max_tokens≥512`（见 §5 提示）。
3. 确认边为 `rel1→alert1→evidence1→vlm1→report1` 且 `alert1→draw1→sink1`。
4. `τ` 与 `cooldown_s` 按误报率调；先小流量观察 `vlm_review/decisions.jsonl`。
5. **其它告警接入**：任何管道只要接上 `alert → evidence → vlm_gate → report` 即自动核验；
   未配 `prompts` 的类型按 `unmatched_policy`（默认 auto，用 alert_name 生成问题）处理，
   要更准再补一条 `prompts[alert_name].question`。

---

## 8. mock 接收端（联调用）

`tools/mock/mock_alert_server.py`（纯标准库）同时提供告警接收与 VLM 桩：

```bash
python3 tools/mock/mock_alert_server.py            # :8902 收告警, :8901 VLM 桩
python3 tools/mock/mock_alert_server.py --alert-only
python3 tools/mock/mock_alert_server.py --vlm-only
```

环境变量控制，用来验证 gate 三分支（不用改代码）：

| 验证 | 设置 | 预期 |
|---|---|---|
| PUSH→上报 | `MOCK_VERDICT=true MOCK_CONF=0.9` | 接收端收到告警（含 base64 图，自动存 `mock_alert_images/`） |
| 判假→DROP | `MOCK_VERDICT=false` | 接收端收不到 |
| 低置信→HOLD | `MOCK_VERDICT=true MOCK_CONF=0.3`（<τ） | 收不到 |
| 超时→HOLD | `MOCK_DELAY_MS=5000`（>timeout） | 收不到，gate 记 error/HOLD |

告警逐条写 `mock_alerts.jsonl`（带 `received_at`），并把上传的 base64 图还原成 jpg，
方便肉眼确认"上报附带的就是那张标注图"。

---

## 9. 注意 / 已知点

- `config/` 与 `models/` 被 gitignore：管道配置是本地模板，不进 git；进 git 的是节点代码/接口/CMake 与 `tools/mock`。
- 未跟踪的旧权重（sevncevision/videomae 等）是普通 blob，**别用全局 `*.pt/*.pth` LFS 规则**
  追溯它们（会产生"待迁移"的假改动）；`.gitattributes` 里对新权重用精确路径。
- gate/report 在 QueuedNode 的单一 worker 线程里跑，libcurl 阻塞调用不卡主链；
  高并发时靠 `queue` 背压 + `drop_newest`，或后续把 VLM 调用移出 worker。
- VLM 只做真假（不纠正类别/谓词）；谓词纠正、多帧时序是后续能力，当前不依赖。
- `ground_truth` 字段默认 null，人工回填后即可统计漏报(应报未报)/误报，作为调 `τ` 和
  后续微调的评测集来源。

## 相关文件

| 文件 | 作用 |
|---|---|
| `include/ai_stream/nodes/i_vlm_gate_node.h` | gate 接口 + `VlmGateConfig` |
| `include/ai_stream/nodes/i_report_node.h` | report 接口 + `ReportConfig` |
| `utils/http_util.h` | header-only libcurl POST + base64 + `${ENV}`（namespace `ai_stream::utils::http`） |
| `src/nodes/gate/vlm_gate_node.{h,cpp}` | 闸门节点（核验、决策、审计、去重） |
| `src/nodes/gate/CMakeLists.txt` | `gate_node` OBJECT 库 |
| `src/nodes/report/report_sink_node.{h,cpp}` | 上报节点（payload + base64 附图） |
| `src/nodes/report/CMakeLists.txt` | `report_node` OBJECT 库 |
| `src/nodes/evidence/evidence_node.{h,cpp}` | 落快照→回挂 `snapshot_path`→转发 |
| `tools/mock/mock_alert_server.py` | 联调用 mock 接收端 + VLM 桩 |
| `config/pipelines/child_near_boundary_pipeline.json` | 含 `vlm1/report1`（默认关）|
