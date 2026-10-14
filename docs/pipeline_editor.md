# 流水线可视化编辑器（Pipeline Editor）

浏览器里以 **DAG 拖拽连线** 的方式生成流水线配置，并可一键 **Build / Start / Stop / Delete**
调用 `http_server` 的 API。技术栈：Vite + React + TypeScript + React Flow。

代码位置：`web/pipeline-editor/`

---

## 1. 快速开始

### 生产（同源，推荐）
```bash
cd web/pipeline-editor
npm install
npm run build          # 产出 dist/
cd ../..
# 起 API 服务，顺带同源托管编辑器
AI_STREAM_WEB_ROOT=./web/pipeline-editor/dist ./build/src/http/http_server 0.0.0.0 8080
```
浏览器打开 `http://<主机IP>:8080/` 即是编辑器；它与 API 同源，**无跨域问题**。

### 开发（热更新）
```bash
cd web/pipeline-editor && npm run dev     # 默认 http://localhost:5173
```
`vite.config.ts` 已把 `/api`、`/health` 代理到 `http://127.0.0.1:8080`，所以本地另起
`http_server`（默认 8080）即可联调，无需配 CORS。

---

## 2. 界面与操作

```
┌──────────┬───────────────────────────┬──────────────┐
│ 节点库    │        DAG 画布            │  属性面板     │
│(按类别分组)│ 拖拽/连线/缩放/小地图       │ 选中节点配参  │
├──────────┴───────────────────────────┴──────────────┤
│ 顶栏: id · API base · Token · 导入/导出/校验/保存 · Build/Start/Stop/Delete/List │
└──────────────────────────────────────────────────────┘
```

- **加节点**：左侧点击节点类型 → 落到画布（自动编号 id）。
- **连线**：从节点右侧圆点拖到目标节点左侧圆点，生成**带方向**的边（源→汇）。
- **选中/删除**：点节点→右侧编辑；选中节点/边按 Delete 删除。
- **配参**：右侧按该节点类型的 schema 渲染表单；复杂结构（`alert.rules`、
  `vlm_gate.prompts`、`evidence.video`、`fusion.class_offsets` 等）用"原始 JSON 编辑"兜底。
- **校验**：唯一 id、未知类型、必填项、边引用有效、**DAG 环检测**、孤立/无源/无汇提示。
- **导出/导入**：导出为后端可直接吃的 JSON；导入还原成图（含布局坐标）。
- **保存到服务器**：`POST /api/v1/pipeline/save-config`，写到
  `<AI_STREAM_STATE_DIR>/pipelines/<id>.json`。

---

## 3. 与后端 API 的对应

| 编辑器按钮 | 接口 | 说明 |
|---|---|---|
| Build | `POST /api/v1/pipeline/build` | 提交整个 `{id, graph}`；异步模式返回 202 |
| Start | `POST /api/v1/pipeline/start` `{id}` | |
| Stop | `POST /api/v1/pipeline/stop` `{id}` | |
| Delete | `DELETE /api/v1/pipeline/delete` `{id}` | |
| List | `GET /api/v1/pipeline/list` | |
| 状态轮询 | `POST /api/v1/pipeline/status` `{id}` | Build/Start 后自动轮询 loading/running |
| 保存到服务器 | `POST /api/v1/pipeline/save-config` | |

- **异步模式**：Build/Start 后编辑器会轮询 `status` 直到 `running/stopped/load_failed`，
  结果打在右下角日志区。
- **鉴权**：若服务端设了 `AI_STREAM_API_TOKEN`，在顶栏 Token 填入（会存 localStorage），
  请求自动带 `Authorization: Bearer`。
- **API base**：同源部署留空即可；跨源调试可填完整地址（需服务端开 CORS）。

---

## 4. 数据模型（导出/提交的就是这个）

```jsonc
{
  "id": "my_pipeline",
  "graph": {
    "nodes": [
      { "id": "rtsp_source_1", "type": "rtsp_source",
        "params": { "url": "rtsp://...", "skip_frames": 1 },
        "position": { "x": 120, "y": 80 } }
    ],
    "edges": [ { "from": "rtsp_source_1", "to": "ffmpeg_decode_1" } ]
  }
}
```
- `position` 仅供编辑器还原布局，后端 `buildFromJson` 会忽略。
- `params` 的层级与结构须与目标节点 `configure()` 读取的键一致。

---

## 5. 节点类型与参数

支持的类型（见 `src/schema/nodeSchema.ts`，与后端注册节点一一对应）：
`rtsp_source` `file_source` `ffmpeg_decode` `resize_normalize` `detection_infer`
`pose_infer` `action_recognition_videomae` `relation_recognition` `fusion` `tracker`
`alert` `evidence` `osd_draw` `rtmp_sink` `mp4_save` `vlm_gate` `report`。

> **新增/改动节点参数时，要同步更新 `src/schema/nodeSchema.ts`**，否则右侧表单/校验
> 与后端不一致。schema 里字段用点路径表示嵌套（如 `detector_config.model_path`），
> 复杂对象用 `json` 类型。

---

## 6. 构建产物与提交

- `web/pipeline-editor/node_modules/` 与 `dist/` 已在 `web/pipeline-editor/.gitignore` 忽略，
  不入库；仓库只跟踪源码。
- 部署时按需 `npm run build` 生成 `dist/`，用 `AI_STREAM_WEB_ROOT` 指过去即可。

---

## 7. 已知限制 / 后续

- 复杂参数目前靠"原始 JSON 编辑"兜底；专用子编辑器（规则逐条表单、prompts 结构化、
  `rule_zones` 多边形在画布上绘制）属后续增强。
- 未做撤销/重做、模板库、从 `config/pipelines/*.json` 反向导入可视化。
