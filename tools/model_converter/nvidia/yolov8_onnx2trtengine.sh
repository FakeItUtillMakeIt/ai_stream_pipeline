#!/bin/bash

# trtexec --onnx=workspace/yolov5s.onnx \
#     --minShapes=images:1x3x640x640 \
#     --maxShapes=images:16x3x640x640 \
#     --optShapes=images:1x3x640x640 \
#     --saveEngine=workspace/yolov5s.engine


cd /sevnce/install/TensorRT-8.6.1.6/bin

./trtexec --onnx=/sevnce/sevncevision/sevncevision/workspace/models/trt/20251217/person_20251203.transd.onnx \
    --minShapes=images:1x3x640x640 \
    --maxShapes=images:16x3x640x640 \
    --optShapes=images:1x3x640x640 \
    --saveEngine=/sevnce/sevncevision/sevncevision/workspace/models/trt/20251217/person_20251203.transd.nx.engine \
    --fp16

# trtexec --onnx=workspace/yolov8n-seg.b1.transd.onnx \
#     --saveEngine=workspace/yolov8n-seg.b1.transd.engine

# ./trtexec --onnx=/sevnce/sevncevision/workspace/models/trt/20240108/20240108.transd.onnx \
#     --saveEngine=/sevnce/sevncevision/workspace/models/trt/20240108/20240108.transd.jetson.half.b1.engine \
#     --fp16

# ./trtexec --onnx=/home/sevnce/sevnceVision/sevnceVison1.0/workspace/models/yolov8n.transd.onnx \
#     --minShapes=images:1x3x640x640 \
#     --maxShapes=images:16x3x640x640 \
#     --optShapes=images:1x3x640x640 \
#     --saveEngine=/home/sevnce/sevnceVision/sevnceVison1.0/workspace/models/yolov8n.transd.fp16.engine \
#     --fp16

# ./trtexec --onnx=/media/yugang/D/AI边缘算法/code/参考代码/trt_yolov8_infer_example/trt_yolov8_infer_example-main/resource/models/yolov8n-seg.b1.transd.onnx \
#     --saveEngine=/media/yugang/D/AI边缘算法/code/参考代码/trt_yolov8_infer_example/trt_yolov8_infer_example-main/workspace/yolov8n-seg.b1.transd.engine
