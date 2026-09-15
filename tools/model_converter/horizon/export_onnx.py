#!/usr/bin/env python3
# ============================================================================
# PyTorch (.pt) -> ONNX 导出（Horizon BPU 专用）
#
# 重要：Horizon hb_compile 需要【静态形状】ONNX。
#   - 必须 dynamic=False（动态形状会导致校准/编译阶段 DFL Reshape 报错，
#     例如 "input_shape_size == size was false ... {1,64,67200} vs {1,4,16,8400}"）
#   - 必须指定 imgsz，与板端模型输入一致（本项目为 640）
#
# 用法:
#   python3 export_onnx.py --pt yolov8s.pt --imgsz 640
# ============================================================================
import argparse
from ultralytics import YOLO


def main() -> None:
    p = argparse.ArgumentParser(description="Export YOLO .pt to static ONNX for Horizon BPU")
    p.add_argument('--pt', required=True, help='输入 PyTorch 权重 (.pt)')
    p.add_argument('--imgsz', type=int, default=640, help='输入边长（需与板端一致）')
    p.add_argument('--opset', type=int, default=12, help='ONNX opset')
    a = p.parse_args()

    model = YOLO(a.pt)
    # dynamic=False：导出静态 [1,3,imgsz,imgsz] 输入
    path = model.export(format='onnx', dynamic=False, opset=a.opset, imgsz=a.imgsz)
    print(f"[INFO] exported static ONNX: {path}")


if __name__ == '__main__':
    main()
