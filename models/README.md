# models/relateanything 与 models/yoloe 说明

这两个目录是「儿童靠近墙/大门」链路（`config/pipelines/child_near_boundary_pipeline.json`）
的模型资产。引擎文件（`.engine`）**不入库**，需要在本机或目标机上按下面的命令生成。

> 本文件只讲**模型资产与导出格式**。管道接线（evidence 是终端、不能串在 draw 前）、
> 规则告警的 duration/容错调参、关系谓词选择，见 `docs/relation_chain.md`。

---

## models/yoloe

| 文件 | 是否入库 | 说明 |
|---|---|---|
| `yoloe_v8s_child_wall_gate.onnx` | 否（大文件） | 固定三类 `child`/`wall`/`gate`，640×640 |
| `yoloe_v8s_child_wall_gate.names.json` | 是 | 类名 sidecar，通道顺序的权威来源 |
| `yoloe_v8s_child_wall_gate.engine` | 否 | 需 `trtexec` 生成，**与 GPU 架构和 TensorRT 版本绑定** |

### 重新导出

```bash
python tools/model_converter/nvidia/export_yoloe_e2e.py \
    --weight /path/to/yoloe-v8s-seg.pt \
    --classes child wall gate \
    --out models/yoloe/yoloe_v8s_child_wall_gate.onnx
```

### 为什么需要这个包装脚本

仓库里已有的检测模型（`models/sevncevision/*_nms.onnx`）输出的是 5 个张量，
`TensorRTDetectionEngine` 按名字取数据（`detection_infer.cpp:159` 起）：

```
det_boxes     (N, 4)   float32   xyxy，已在 640 输入坐标系内
det_scores    (N,)     float32
det_classes   (N, 1)   int64
det_batch_ids (N, 1)   int64
det_num_dets  ()       int64
```

ultralytics 的 `export(nms=True)` 给的是打包张量 `output0 (1, max_det, 4+nc+nm)`，
直接塞进去取不到数据。脚本把它拆成上面 5 个输出并补齐 metadata。

包装时最关键的格式约定，改动务必注意：

**`det_boxes` 必须是 `cxcywh`，不是 `xyxy`。** `detection_infer` 的
`postprocessBatch`（detection_infer.cpp:653-671）把每行读作 `cx,cy,w,h` 并做
`box.x = orig_cx - orig_w/2`。ultralytics 的 NMS 输出是 xyxy，直接塞进去会被
曲解成"偏移 + 宽高翻倍"的框：不报错，但框画到画面外（`draw` 里表现为 child
框看不见，只有 wall 那种本来就贴近边界的还能露一角），而且关系模型拿到的几何
也是错的。现有 `*_nms.onnx` 输出的就是 cxcywh，导出脚本已把 xyxy 转成 cxcywh。
修好之后 test5 之外的真实视频关系分从 ~0.2 提到 ~0.43，印证之前是喂了错几何。

其余两个细节：

- `det_scores` 必须是**一维每行最大分**，因为引擎按 `scores[i]` 消费
  （`detection_infer.cpp:567`），不是 `(N, nc)` 的分数表。
- 分割权重的 `output0` 最后一维是 `4 + nc + nm`（本例 38），Reshape 必须用
  这个**真实维度**，用 `4 + nc`（7）会直接形状不兼容。mask 系数段切出来丢掉。

### 关键约束：类别顺序

`set_classes()` 之后的文本嵌入已烤进计算图，**engine 不可再改类别**。
通道顺序严格等于导出顺序：

```python
set_classes(["child", "wall", "gate"])   # 通道 4/5/6
```

顺序写反不会报错，只会静默标签错位。改类别必须重新导出，
并同步改 `detector_config.model_class_names`。

---

## models/relateanything

| 文件 | 是否入库 | 说明 |
|---|---|---|
| `relateanything.onnx` | 否（198MB） | RelateAnything 图，448×448 |
| `predicate_bank.json` | 是 | 243 谓词 + `W`(243×512) + `alpha`(243) + 标定 |
| `relateanything.json` | 是 | 输入尺寸、`a`/`b` 标定、输出契约 |
| `calibration.json` | 是 | 标定来源与指标（ECE/AUC） |
| `thresholds.json` | 是 | bank 逐谓词阈值，**仅供查表，默认不启用** |
| `relateanything.engine` | 否 | 需转换，绑定 GPU 架构与 TensorRT 版本 |

### 图契约

```
输入  image       [batch, 3, 448, 448]
      boxes       [batch, num_boxes, 4]      xyxy，已在 448 输入坐标系内
      box_counts  [batch]
      W           [num_predicates, 512]
      alpha       [num_predicates]
输出  pred_logits  [batch, num_pairs, num_predicates]
      pair_logits  [batch, num_pairs]
      sub_idx      [batch, num_pairs]
      obj_idx      [batch, num_pairs]
      valid_mask   [batch, num_pairs]
```

分数由标定合成：

```
sigmoid(a * (pred_logit + w * pair_logit) + b)     a=0.5651  b=-1.9623
```

`w` 就是 `W` 对应行，按谓词名回查。

### `num_boxes` 是动态维，TRT 必须设 profile

`box_counts` 每帧随目标数变化，而 `boxes` 第二维是符号名 `num_boxes`。
只用单点 shape 建 profile 会在运行期 shape 不匹配。建引擎时要给区间，例如：

```bash
trtexec --onnx=models/relateanything/relateanything.onnx \
        --saveEngine=models/relateanything/relateanything.engine \
        --minShapes=image:1x3x448x448,boxes:1x2x4,box_counts:1x1,W:243x512,alpha:243 \
        --optShapes=image:1x3x448x448,boxes:1x32x4,box_counts:1x1,W:243x512,alpha:243 \
        --maxShapes=image:1x3x448x448,boxes:1x32x4,box_counts:1x1,W:243x512,alpha:243
```

`boxes` 的下界取 2 是因为关系图要成对；实际节点最多送 `max_boxes` 个框
（与 `relateanything.json` 的 `max_boxes: 32` 一致）。

### `near` 不在默认谓词里

243 个谓词中有 `near`，但它不是默认激活项，且 `thresholds.json` 给它的
阈值高达 0.965。实测（`DART/test_img/test5.jpg`，儿童站砖墙前）：

```
beside 0.467   behind 0.213   at 0.211   near / next to  无输出
```

所以当前默认谓词用 `beside`、`in front of`、`behind`、`on`。

**不要**用 bank 的逐谓词阈值去覆盖配置：`beside 0.980`、`in front of 0.870`
会压死实测约 0.4 的关系分。以配置里的 `score_threshold` 为主，
`per_predicate_threshold` 只在明确知道后果时才设。

### gate 的谓词尚未验证

没有真实的「儿童靠近 gate」样本，`gate` 该用 `beside` 还是 `in front of`
目前是推测。首次跑通后要用真实素材标定。