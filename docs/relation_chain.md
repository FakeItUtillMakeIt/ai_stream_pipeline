# 儿童靠近边界（关系检测）

`child` 与 `wall`/`gate` 的空间关系判断。检测器用 **YOLOE**（开放词表，固定三类），
关系模型用 **RelateAnything** 场景图，规则把两者的结果收敛成告警。

## 为什么检测器要换

原有主模型（`sevncevision` 18 类 PPE/行为）里**没有 `wall`、`gate`，也没有 `child`**，
关系模型没有框可打分。YOLOE 可以用提示词指定类别，导出时把类别嵌入烤进模型头，
推理时不再需要文本编码器。

## 数据流

```
rtsp_source → ffmpeg_decode → resize_normalize(640) → detection_infer → tracker
            → relation_recognition → alert
     alert ─┬→ osd_draw ─┬→ rtmp_sink          # 实时预览，不受核验影响
            │            └→ evidence
            └→ evidence → [vlm_gate →] report  # 告警支路，可选过 VLM 闸门
```

`relation_recognition` 消费 tracker 之后的 `InferenceResultPacket`：
从 `source_frame->source_mat` 取原图，按 448 做 letterbox，把 detections 映射到
同一坐标系，一次前向拿到全部三元组写入 `packet->relations`。

### 接线：evidence 收两路，且只在告警支路下沉转发

`evidence` 同时收 `alert`(META_DATA 告警事件) 与 `draw`(DECODED_FRAME 已画框帧)。
它**不透传实时帧流**——`DECODED_FRAME` 一律吞掉，所以**绝不能把 evidence 串在
`draw → sink` 中间**，否则 draw/sink 收不到帧、快照目录空、RTMP 无输出且**不报错**。

但 evidence 在**落了一张告警快照后**，会把携带该告警的 `META_DATA` 包（并回挂
`extra_data["snapshot_path"]`）`broadcast` 给下游，供 `vlm_gate` 取标注图核验、
或无 gate 时直连 `report` 上报。也就是说：evidence 只在"有告警快照"这一个时机转发
**告警事件**，不转发普通帧。

```
alert → draw → sink            # 主链，帧一路传到推流
alert → evidence               # 告警事件 → 触发录制/快照
draw  → evidence               # 绘制后的帧 → 预录缓冲、证据截图（不转发）
evidence → vlm_gate → report   # 告警落快照后转发该事件 → 核验 → 上报
```

告警核验与上报（base64 附图）详见 `docs/vlm_alert_gate.md`。

## 坐标系：640 与 448 不会混用

这是最容易搞错的地方，所以把整条链路的坐标空间列清楚：

| 环节 | `mat` | `source_mat` | detections |
|---|---|---|---|
| `ffmpeg_decode` | 解码原图 | 解码原图（同一个） | — |
| `resize_normalize(640)` | **变成 640** | **仍是原图** | — |
| `detection_infer` | 同上 | 同上 | **反变换回原图坐标** |
| `tracker` | 同上 | 同上 | 原地改 `track_id` |
| `relation_recognition` | — | **只用这个**（原图） | 当成原图坐标用 |

关键两行：

```cpp
// detection_infer.cpp:662 —— 检测框已被还原到原图像素坐标
float inv_scale = 1.0f / letter_scale[batch_id];
float orig_cx = (cx - pad_x) * inv_scale;
```
```cpp
// resize_normalize.cpp:294-295
new_packet->mat = processed_mat;                        // <- 这个是 640，别用
new_packet->source_mat = frame->source_mat;             // <- 这个永远是原图
```

所以关系节点拿 `source_frame->source_mat`（原图）+ 原图坐标的框，
**自己算 448 的 scale/pad** 重新映射一次。640 全程不参与。

> 误用 `mat` 才是真正的混用：拿到 640 图 + 原图坐标的框，两边差一个缩放系数。
> 这种错**不会报错**，只表现为关系分莫名偏低。

节点里为此加了三道防护：

1. `source_mat` 空指针 / 空图检查（`ffmpeg_decode.cpp:438` 无条件赋值，但 NV12 与
   GPU 分支是追加而非清空，仍按 `pose_infer.cpp:138` 的同一标准检查）；
2. 框按 448 画布做越界自检——若整体越界或整体为零，直接报错停下，
   并提示"detections 可能不在原图坐标"，而不是继续跑出一堆假关系；
3. 日志里打印 `boxes` 与 `relations` 数量，便于现场核对。

`tracker` 是原地修改同一个 packet 再广播，所以 `source_frame` 与 detections
保证同帧，不会出现"A 帧的图配 B 帧的框"。

### 引擎的 `det_boxes` 必须是 cxcywh，不是 xyxy

`detection_infer.cpp:653-671` 把 `det_boxes` 前三列当 `cx,cy` 解析
（`box.x = orig_cx - orig_w/2`），即**期望 cxcywh**。仓库里现有的 `*_nms.onnx`
输出的就是 cxcywh。若引擎吐 xyxy（比如直接塞 ultralytics 的 NMS 输出），会被曲解成
"坐标偏移 + 宽高翻倍"：框跑到画面外、`draw` 里看不见、关系模型拿到错误几何，但**不报错**——
只表现为"有的框看得见有的看不见"和"关系分莫名偏低"。`tools/model_converter/nvidia/export_yoloe_e2e.py`
已把这层转成 cxcywh，自定义导出时务必对齐。

## 节点参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `model_path` | 必填 | `.engine` 或 `.onnx` |
| `predicate_bank` | 空 | 谓词表 JSON，由 `export_predicate_bank.py` 生成 |
| `img_size` | `0`（用模型自带） | 关系图边长，RelateAnything 是 448 |
| `max_boxes` | `0`（用模型自带） | 每帧最多送入的框数 |
| `predicates` | 引擎默认集 | 运行期激活的谓词，**换谓词不用重新导出引擎** |
| `score_threshold` | `0.2` | 关系分门限 |
| `box_confidence` | `0.25` | 低于此值的框不送入关系模型 |
| `boundary_classes` | `["wall","gate"]` | 哪些类别算"边界" |
| `per_predicate_threshold` | 空 | 逐谓词覆盖，见下方警告 |
| `max_relations` | `256` | 单帧返回上限 |
| `backend` | `auto` | `tensorrt` / `onnxruntime` / `cpu` |

### 不要照抄模型 bank 里的逐谓词阈值

`predicate_bank.json` 里有每个谓词的标定工作点：

```
beside 0.980    in front of 0.870    behind 0.900    on 0.935
```

**这些值不能直接用作门限。** 实测关系分常在 0.4 量级（`beside` 在"儿童站砖墙前"
那张图上最高 0.467），一旦让 0.980 生效，规则会**永远静默且不报任何错**。
`score_threshold` 才是你要调的旋钮；`per_predicate_threshold` 只在"某个谓词太吵、
想单独压它"时才用。

## 规则参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `name` | 类型名 | 告警名 |
| `rule_zones` | `[]` | 空 = 全域监测；配了多边形 = 只在区内判 |
| `child_class` | `child` | 主体类别 |
| `boundary_classes` | `["wall","gate"]` | 客体类别 |
| `relation_threshold` | `0.2` | 关系分门限（规则侧再收一次口） |
| `duration_ms` | `1000` | 命中需**累计**超过此时长才从 DEFAULT 进 OCCUR（也接受键名 `alert_duration_ms`） |
| `max_disappear_count` | 基类 5 | 允许连续多少帧没命中仍保留事件，超过则 erase、`detect_ms` 归零 |
| `geometric_fallback` | `false` | 见下 |
| `max_pixel_gap` | `60` | 几何回退的像素间距 |

### "命中"不等于"告警"：状态机怎么攒 duration

`rule_logic` 打印 `child near wall score=…` 只代表这一帧有命中，
真正告警要过 `alert_rule_base.h` 的状态机：`updateZoneEvent` 累计
`duration_ms = 当前命中时刻 - 首次命中时刻`，**只有 `duration_ms > alert_duration_ms_`
才把状态从 DEFAULT 推到 OCCUR**，OCCUR 才进 `alert_events`、evidence 才录制/截图。

踩过的坑：关系分在阈值附近反复穿越（`0.248, 0.229, 0.194, 0.189, 0.201…`），
命中是断续的。默认 `max_disappear_count=5`（约 0.2s）一断就 erase、`detect_ms` 归零，
`duration_ms` 永远攒不到 1000，于是**规则每帧都"命中"却从不告警**。
两个修法配合用：

- `relation_threshold` 下调到贴合模型工作点（本场景实测稳定命中约 **0.15**，
  框几何修对后 `in front of` 能到 0.4+）；
- `max_disappear_count` 放宽（如 **30** ≈ 1.2s），桥接 child 偶尔漏检 / 关系分抖动。

调参用 `child_on_wall.mp4` 这类"持续贴近"的真实素材验证 duration 能否越过阈值。

### 方向必须显式判

关系是有向的三元组，模型可能给出「墙 beside 儿童」。规则只认
**主体=child 且客体∈boundary**，只看谓词会把两者混为一谈。

### 几何回退默认关闭

`geometric_fallback: false` 是刻意的：开启后关系模型没跑起来（比如引擎缺失、
`predicates` 全写错）会退化成纯距离判定，看起来"一切正常"。
那比直接不告警更危险——你不会知道模型其实没在工作。

## 谓词选择

实测（`DART/test_img/test5.jpg`，儿童站在砖墙前）：

| 谓词 | 最高分 | 数量 | 是否在模型默认激活集 |
|---|---|---|---|
| `beside` | **0.467** | 16 | 是 |
| `behind` | 0.213 | 8 | 是 |
| `at` | 0.211 | 4 | 否 |
| `near` | 无输出 | 0 | **否** |
| `next to` | 无输出 | 0 | **否** |

`near` 和 `next to` **在 243 谓词库里，可以用**（`vocab_mode: input`，`W`/`alpha`
是图输入，运行时传对应行即可激活），但逐谓词阈值分别是 0.965 / 0.940，
而这张图上所有关系分最高才 0.47——差一个数量级，实测一条都出不来。

所以默认用 `beside` + `in front of` + `behind` + `on`。
**`gate` 的谓词组合没有验证过**（手上没有任何带门的样本），需要真实画面确认。

## 模型准备

> **原始模型已放进对应目录**（`models/` 被 gitignore，只在本机留存）：
>
> | 目录 | 源权重 | 用途 |
> |---|---|---|
> | `models/yoloe/` | `yoloe-v8s-seg.pt` | 导出 YOLOE ONNX 的源权重 |
> | `models/relateanything/` | `model.pth` + `text_student.pt` + `predicate_embeddings.npz` | 导出 RelateAnything ONNX 的源权重 |
>
> 源 `.pt/.pth` 已就位，但**从源权重重导 ONNX** 的 `export_onnx.py` 需要完整训练环境
> （ultralytics + relsgg + HF 依赖）；本仓库通常直接复用已导出的 `.onnx`。

### 1. YOLOE 导出（固定三类）

用 `tools/model_converter/nvidia/export_yoloe_e2e.py`，它会导出端到端 NMS 的图，
并**包装成引擎需要的 5 输出格式**（`det_boxes`(cxcywh)/`det_scores`/`det_classes`/
`det_batch_ids`/`det_num_dets`），详见 `models/README.md`：

```bash
# pt -> onnx
python tools/model_converter/nvidia/export_yoloe_e2e.py \
    --weight models/yoloe/yoloe-v8s-seg.pt \
    --classes child wall gate \
    --out models/yoloe/yoloe_v8s_child_wall_gate.onnx
```

> `yoloe-v8s.pt` 在 ultralytics 8.4.x **并不存在**，promptable YOLOE 只以 `-seg` /
> `-seg-pf` 变体发布，所以源权重是 `yoloe-v8s-seg.pt`（就是"YOLOE v8s"）。
> **类别顺序 = 通道顺序**，`set_classes` 后文本嵌入烤进图里、engine 不可再改类别；
> `detector_config.model_class_names` 必须与导出顺序一致，否则静默标签错位。
> sidecar `*.names.json` 是这个顺序的权威来源，别手抄。

### 2. 谓词表导出

```bash
cd /path/to/RelateAnything
python deploy/export_predicate_bank.py \
    --dist deploy/dist/relsgg-vits16plus \
    --out /path/to/ai_stream_pipeline/models/relateanything/predicate_bank.json \
    --predicates beside "in front of" behind on
```

### 3. ONNX → Engine（统一用脚本）

两条链路的 `.onnx -> .engine` 都用 `tools/model_converter/nvidia/build_engines.sh`，
参数已固化，避免手敲漏 profile：

```bash
bash tools/model_converter/nvidia/build_engines.sh all           # 两个都转
bash tools/model_converter/nvidia/build_engines.sh yoloe         # 只转 YOLOE
bash tools/model_converter/nvidia/build_engines.sh relateanything # 只转关系
```

展开就是：

```bash
# YOLOE：batch 固定 1，输出动态维 TRT 自处理，不需要 profile
trtexec --onnx=models/yoloe/yoloe_v8s_child_wall_gate.onnx \
        --saveEngine=models/yoloe/yoloe_v8s_child_wall_gate.engine \
        --fp16 --builderOptimizationLevel=5

# RelateAnything：num_boxes 是动态维，必须给 min/opt/max profile
trtexec --onnx=models/relateanything/relateanything.onnx \
        --saveEngine=models/relateanything/relateanything.engine --fp16 \
        --minShapes=image:1x3x448x448,boxes:1x2x4,box_counts:1,W:243x512,alpha:243 \
        --optShapes=image:1x3x448x448,boxes:1x32x4,box_counts:1,W:243x512,alpha:243 \
        --maxShapes=image:1x3x448x448,boxes:1x32x4,box_counts:1,W:243x512,alpha:243
```

**关键坑（脚本里已处理，改脚本别破坏）：**

- 引擎**必须在目标 GPU 上构建**，绑定架构 + TensorRT 版本，换卡重跑脚本即可。
- profile 里 `box_counts:1`、`alpha:243` 必须是 **1-D**；写成 `1x1` / `243x1`
  会被 TRT 拒（`profile 0 has 2 dimensions`）。
- `boxes` 的 `num_boxes` 是动态维，`min/opt/max` 三档必须给全，否则运行期
  `setInputShape` 才报错。
- `VOCAB=243` / `DIM=512` / `MAX_BOXES=32` 要与 `predicate_bank.json`、
  `relateanything.json` 一致；换谓词库时同步改脚本顶部变量。
- 默认 `--fp16`。要复现数值/换弱卡，去掉 `--fp16` 用 FP32（更慢、更稳）。

## 已知限制

- **YOLOE 用端到端 NMS 导出**，`det_scores` 是每行最大分的一维 `(N,)`
  （引擎按 `scores[i]` 消费，`detection_infer.cpp:567`），`det_boxes` 是 cxcywh。
  不再需要早期"按 `model_class_names.size()` 把 39 通道截断"那套——那是原始
  （非 NMS）导出才有的问题，`export_yoloe_e2e.py` 已用 Split 丢掉 mask 系数段。
- **`pred_logits` 的谓词维是静态 243，不是激活数。** `W`/`alpha` 虽作为图输入，
  但 `num_predicates` 在导出时是固定维（=整张词表），运行期不能按激活数 `setInputShape`
  （TRT 会报 `Static dimension mismatch`）。做法：`W`/`alpha` 填满整表、未激活行填 0，
  host 解码时再用 `only_predicates` 过滤——否则会把从未启用的谓词当成结果吐出来。
- **候选配对数 ≠ 框数**：`boxes=32` 时图输出 `pairs=128`，host 缓冲必须按引擎
  `pred_logits` 的真实维度分配，按框数猜会读到越界垃圾下标（表现为"推理成功但 0 条关系"）。
- **`sub_idx`/`obj_idx` 是 int64、`valid_mask` 是 bool**，按 int32/float 读会静默错位。
- **引擎绑 GPU 架构 + TensorRT 版本**，换卡必须重新构建。
- 关系模型约 13–16ms/图（TRT FP16, ViT-S+），YOLOE ~17ms/张（预热后）。
- 关系后端目前**只有 TensorRT（x86）**，无 RKNN/ONNXRuntime 实现；RK3588 需另做后端。
- `gate` 的谓词组合仍无真实样本验证。

## 文件

| 文件 | 作用 |
|---|---|
| `include/ai_stream/hal/i_relation.h` | 关系引擎 HAL 接口 |
| `include/ai_stream/hal/relation_factory.h` | 后端工厂 + 注册宏 |
| `src/hal/relation_factory.cpp` | 工厂实现（AUTO 选择顺序 TensorRT > ONNXRuntime > CPU） |
| `src/hal/infer/common/relation_decode.{h,cpp}` | 解码器，逐行对应 `deploy/postprocess.py` |
| `src/hal/infer/nvidia/tensorrt_relation.{h,cpp}` | TensorRT 后端 |
| `include/ai_stream/nodes/i_relation_node.h` | 节点接口 |
| `src/nodes/infer/relation_recognition.{h,cpp}` | 节点实现 |
| `src/rules/alert/child_near_boundary_rule.{h,cpp}` | 规则 |
| `config/pipelines/child_near_boundary_pipeline.json` | 管道示例 |
| `tools/model_converter/nvidia/export_yoloe_e2e.py` | YOLOE 导出并包装成引擎的 5 输出（cxcywh） |
| `tests/unit/nodes/test_relation_decode.cpp` | 解码 host 端回归（不依赖 GPU/模型） |
| `tests/integration/test_relation_cpp.cpp` | 真实推理测试（需 GPU+engine，不进 ctest） |
| `models/README.md` | 模型资产、导出、engine 构建与格式约定 |

`packet.h` 新增：`AlertType::CHILD_NEAR_BOUNDARY = 20`、
`InferenceResultPacket::RelationResult` 与 `relations` 字段。

---

## 模型文件位置

资产已就位，engine 需在目标机生成：

| 目录 | 内容 |
|---|---|
| `models/yoloe/` | `yoloe_v8s_child_wall_gate.onnx` + `.names.json` |
| `models/relateanything/` | `relateanything.onnx`、`predicate_bank.json`、`relateanything.json`、`calibration.json`、`thresholds.json` |

详见 `models/README.md`，其中记录了：

- YOLOE 为什么需要 `tools/model_converter/nvidia/export_yoloe_e2e.py` 包装（引擎按
  tensor 名取 5 个输出，ultralytics 给的是打包张量）
- **`det_boxes` 必须是 cxcywh**（引擎按 cx,cy,w,h 解析），xyxy 会被曲解成偏移+翻倍的框
- 类别顺序 = 通道顺序，engine 不可改类别
- 关系 ONNX 的 `num_boxes` 是动态维，`trtexec` 必须给 profile 区间，否则运行期 shape 不匹配
- `near` 的实测分数与为何不用 bank 的逐谓词阈值

上线前务必对照本文的**接线**（evidence 只在告警支路转发标注事件、不透传实时帧，
不能串在 `draw → sink` 之间）与**规则 duration**
（`relation_threshold` 调低 + `max_disappear_count` 放宽，否则断续命中攒不到时长、永不告警）。
