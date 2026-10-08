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
            → relation_recognition → alert → evidence → osd_draw → rtmp_sink
```

`relation_recognition` 消费 tracker 之后的 `InferenceResultPacket`：
从 `source_frame->source_mat` 取原图，按 448 做 letterbox，把 detections 映射到
同一坐标系，一次前向拿到全部三元组写入 `packet->relations`。

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
| `geometric_fallback` | `false` | 见下 |
| `max_pixel_gap` | `60` | 几何回退的像素间距 |

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

### 1. YOLOE 导出（固定三类）

```bash
python -c "
from ultralytics import YOLOE
m = YOLOE('yoloe-v8s.pt')
m.set_classes(['child', 'wall', 'gate'])          # 顺序即通道顺序
m.export(format='onnx', imgsz=640, opset=17, simplify=True, nms=False, batch=1)"
# 再用 tools/model_converter/nvidia 转成 .engine
```

> **通道顺序 = 导出时的提示顺序**，而 `detector_config.model_class_names` 是手写数组。
> 两者不一致不会报错，只是标签整体错位（框还在、名字反了）。
> 建议加测试断言二者一致，别靠人记。

### 2. 谓词表导出

```bash
cd /path/to/RelateAnything
python deploy/export_predicate_bank.py \
    --dist deploy/dist/relsgg-vits16plus \
    --out /path/to/ai_stream_pipeline/models/relation/predicate_bank.json \
    --predicates beside "in front of" behind on
```

### 3. 关系引擎

```bash
python deploy/export_onnx.py --out relateanything.onnx
python deploy/export_tensorrt.py --onnx relateanything.onnx --engine relateanything.engine
# 引擎必须在**目标 GPU 上**构建
```

## 已知限制

- **YOLOE 导出的 ONNX 有 39 个通道，只有前 3 个有意义。**
  通道 4/5/6（child/wall/gate）是后 sigmoid 的概率，通道 7..34 是未过 sigmoid 的
  原始 logits，最大值到 6.32。`detection_post` 必须按 `model_class_names.size()`
  截断，否则会在不存在的类上产生 0.998 的高置信度框。
  试过把 ONNX 裁成 7 通道，**失败**（onnxruntime 回落到宽松合并，实际输出仍是 39 通道）。
- **`relation` 输出通道数是激活谓词数，不是 243。**
  `W`/`alpha` 作为图输入传入，只有激活的那几列有意义。解码器已按 `predicates.size()` 截断。
- **引擎绑 GPU 架构 + TensorRT 版本**，换卡必须重新构建。
- 关系模型 17.2ms/图（TRT FP32, ViT-S+），YOLOE 17ms/张（预热后），30fps 流逐帧跑约占 50% 算力。

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

`packet.h` 新增：`AlertType::CHILD_NEAR_BOUNDARY = 20`、
`InferenceResultPacket::RelationResult` 与 `relations` 字段。
