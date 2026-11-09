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