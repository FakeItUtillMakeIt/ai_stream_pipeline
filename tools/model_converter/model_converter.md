## 模型转换
### 1.转换带nms的onnx模型
```

    python3 ./tools/model_converter/yolov8_nms_in_graph_batch.py ./models/sevncevision/20260624_sevnce_18cls.onnx /home/sevnce/project/images --convert --output-model /home/sevnce/project/ai_stream_pipeline/models/sevncevision/20260624_sevnce_18cls_nms.onnx --num-classes=18 --conf=0.5 --nms=0.5
```