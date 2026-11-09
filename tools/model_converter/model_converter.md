## 模型转换
### 1.转换带nms的onnx模型
```

    python3 ./tools/model_converter/yolov8_nms_in_graph_batch.py ./models/sevncevision/20260624_sevnce_18cls.onnx /home/sevnce/project/images --convert --output-model /home/sevnce/project/ai_stream_pipeline/models/sevncevision/20260624_sevnce_18cls_nms.onnx --num-classes=18 --conf=0.5 --nms=0.5
```

### 2.Ascend ONNX → OM（ATC）

```bash
# 需先 source CANN set_env.sh（脚本也会自动 source 默认路径）
tools/model_converter/ascend/convert_onnx_to_om.sh \
    models/sevncevision/20260921_sevnce_9_cls_nms_trans_640.onnx \
    images 640 640 Ascend310P3 \
    models/sevncevision/20260921_sevnce_9_cls_nms_trans_640
```

生成的 `.om` 供 `detection_infer` 的 `"backend": "ascend"` 使用；
输出 5 张量契约（det_boxes/det_scores/det_classes/det_batch_ids/det_num_dets）。

### 3.Ascend 动作识别 VideoMAE ONNX → OM

```bash
INPUT_SHAPE="input:1,16,3,224,224" \
tools/model_converter/ascend/convert_onnx_to_om.sh \
    models/action_recognition/action_recognition.onnx \
    input 224 224 Ascend310P3 \
    models/action_recognition/action_recognition
```

VideoMAE 输入是 5D `[1,T,3,H,W]`，脚本会自动省略 `--input_format`；
输出的 `[1,3]` logits 由 `AscendActionRecognition` 做 softmax/argmax。

### 4.Ascend YOLO-Pose ONNX → OM

原始 `yolov8n_pose.onnx` 是 opset 18，直接喂 ATC 会在
`libops_all_onnx_plugin.so` 里段错误。专用脚本会先 `onnxsim` 固定
`[1,3,640,640]`，把 Resize 的 opset-18 属性和 Split 的 sizes 输入降级到
opset 13，再调用 ATC：

```bash
python3 tools/model_converter/ascend/convert_pose_to_om.py \
    models/yolov8n/yolov8n_pose.trans.onnx \
    --output models/yolov8n/yolov8n_pose
```

推荐 `*_pose.trans.onnx`（输出 `[1,8400,56]`）。`.om` 供 `pose_infer`
使用；节点 AUTO 会按 `.om` 扩展名选择 Ascend 后端。

### 5.Ascend RelateAnything ONNX → OM

原始 `relateanything.onnx` 包含 9 个 `If` 子图，ATC 能编译但运行期 shape
inference 报 `[1,1568,0] cannot broadcast`。专用脚本按当前导出语义内联
`If` 分支，用 `onnxsim` 固定 `image/boxes/box_counts/W/alpha`，再调用 ATC：

```bash
python3 tools/model_converter/ascend/convert_relation_to_om.py \
    models/relateanything/relateanything.onnx \
    --output models/relateanything/relateanything
```

生成的 `.om` 供 `relation_recognition` 使用，配置
`"backend": "ascend"` + `"model_path": "*.om"`，谓词表仍用
`predicate_bank.json`。脚本依赖 `onnx` / `onnxsim`（`pip install --user` 即可），
并会自动把 CANN 的 `python/site-packages` 与驱动库路径注入 ATC 子进程。