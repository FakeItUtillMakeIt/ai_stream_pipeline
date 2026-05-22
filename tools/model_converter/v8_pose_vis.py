#!/usr/bin/env python3
"""
YOLOv8-pose Batch 推理脚本（支持多图/目录输入，兼容 decoded 模式）
用法:
    # 单张图（和原来一样）
    python v8_pose_vis_batch.py --model model.onnx --image person.jpg

    # 多张图 batch 推理
    python v8_pose_vis_batch.py --model model.onnx --image img1.jpg img2.jpg img3.jpg

    # 整个目录
    python v8_pose_vis_batch.py --model model.onnx --input-dir ./crops/ --output-dir ./results/

    # 两阶段: 带原图检测框信息（JSON 格式，用于关键点映射回原图）
    python v8_pose_vis_batch.py --model model.onnx --image c1.jpg c2.jpg         --boxes-meta '[{"x1":100,"y1":200,"x2":300,"y2":500}, {"x1":400,...}]'         --orig-size "1920x1080" --output-dir ./results/
"""
import argparse
import cv2
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import os
from pathlib import Path

# ==================== COCO 定义 ====================
KPT_NAMES = ["nose","left_eye","right_eye","left_ear","right_ear",
             "left_shoulder","right_shoulder","left_elbow","right_elbow",
             "left_wrist","right_wrist","left_hip","right_hip",
             "left_knee","right_knee","left_ankle","right_ankle"]

SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(7,9),(6,8),(8,10),
            (5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]

KPT_COLORS = [(0,0,255),(0,255,0),(255,0,0),(0,255,255),(255,0,255),
              (255,255,0),(128,0,0),(0,128,0),(0,0,128),(128,128,0),
              (128,0,128),(0,128,128),(255,128,0),(128,255,0),(0,128,255),
              (255,0,128),(128,0,255)]


def preprocess_batch(image_paths, input_size=640):
    """
    批量预处理
    返回: batch_tensor [B,3,H,W], list of (orig_w, orig_h), list of img_bgr
    """
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    tensors = []
    orig_sizes = []
    img_bgrs = []

    for path in image_paths:
        img = Image.open(path).convert('RGB')
        orig_w, orig_h = img.size
        img_r = img.resize((input_size, input_size), Image.Resampling.LANCZOS)
        arr = np.array(img_r).astype(np.float32) / 255.0
        arr = arr.transpose(2, 0, 1)  # HWC -> CHW
        tensors.append(arr)
        orig_sizes.append((orig_w, orig_h))
        img_bgrs.append(cv2.imread(path))

    batch_tensor = np.stack(tensors, axis=0)  # [B, 3, H, W]
    return batch_tensor, orig_sizes, img_bgrs


def decode_predictions_batch(output, mode="decoded", input_size=640, conf_thres=0.25, kpt_conf_thres=0.5):
    """
    批量解码
    output: [B, N, 56]
    返回: list of dict, 每个 dict 包含单图的最佳结果
    """
    batch_size = output.shape[0]
    results = []

    for b in range(batch_size):
        pred = output[b]  # [N, 56]
        N = pred.shape[0]

        box = pred[:, :4].copy()      # [N, 4]
        score = pred[:, 4:5].copy()   # [N, 1]
        kpt = pred[:, 5:].copy()      # [N, 51]

        if mode == "raw":
            # sigmoid
            box = 1.0 / (1.0 + np.exp(-box))
            score = 1.0 / (1.0 + np.exp(-score))
            kpt[:, 0::3] = 1.0 / (1.0 + np.exp(-kpt[:, 0::3]))
            kpt[:, 1::3] = 1.0 / (1.0 + np.exp(-kpt[:, 1::3]))
            kpt[:, 2::3] = 1.0 / (1.0 + np.exp(-kpt[:, 2::3]))
            # 这里需要 anchors/strides 解码，简化处理：抛出提示
            raise NotImplementedError("raw 模式在 batch 中需要传入 anchors/strides，请用 decoded 模式")

        elif mode == "decoded":
            score = 1.0 / (1.0 + np.exp(-score))
            kpt[:, 2::3] = 1.0 / (1.0 + np.exp(-kpt[:, 2::3]))
            # box, kpt x,y 已是绝对坐标，保持不变

        elif mode == "normalized":
            score = 1.0 / (1.0 + np.exp(-score))
            kpt[:, 2::3] = 1.0 / (1.0 + np.exp(-kpt[:, 2::3]))
            box *= input_size
            kpt[:, 0::3] *= input_size
            kpt[:, 1::3] *= input_size

        # 取置信度最高的一个（单目标/crop 场景）
        # 如果是整图多人，建议改成取所有 score > conf_thres 的，再分别解码
        best_idx = int(np.argmax(score))
        best_score = float(score[best_idx])

        if best_score < conf_thres:
            results.append(None)
            continue

        best_box = box[best_idx]
        best_kpt = kpt[best_idx].reshape(17, 3)

        # 过滤低置信度关键点
        best_kpt[best_kpt[:, 2] < kpt_conf_thres, :2] = 0

        results.append({
            'box': best_box,          # [4] cx,cy,w,h 或 x1,y1,x2,y2 取决于模型
            'score': best_score,
            'keypoints': best_kpt,    # [17, 3]
        })

    return results


def draw_single(img_bgr, result, input_size=640, box_meta=None):
    """
    在单张图上绘制结果
    box_meta: 如果是两阶段 crop 图，传入原图检测框 {"x1":..,"y1":..,"x2":..,"y2":..}
                用于把关键点从 640x640 crop 坐标映射回原图
    """
    if result is None:
        cv2.putText(img_bgr, "No person detected", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        return img_bgr

    h, w = img_bgr.shape[:2]
    box = result['box']
    score = result['score']
    kpts = result['keypoints']

    # 坐标缩放因子
    if box_meta is not None:
        # 两阶段: 关键点在 640x640 crop 内，需要映射回原图
        orig_x1, orig_y1 = box_meta['x1'], box_meta['y1']
        orig_x2, orig_y2 = box_meta['x2'], box_meta['y2']
        scale_x = (orig_x2 - orig_x1) / float(input_size)
        scale_y = (orig_y2 - orig_y1) / float(input_size)
        offset_x, offset_y = orig_x1, orig_y1
        # 框也映射回原图（如果框是 crop 内的相对坐标）
        bx, by, bw, bh = box
        bx = bx * scale_x + offset_x
        by = by * scale_y + offset_y
        bw = bw * scale_x
        bh = bh * scale_y
    else:
        # 单阶段/整图推理: 直接按 input_size 缩放
        scale_x = w / float(input_size)
        scale_y = h / float(input_size)
        offset_x, offset_y = 0, 0
        bx, by, bw, bh = box
        bx *= scale_x
        by *= scale_y
        bw *= scale_x
        bh *= scale_y

    # 画框 (假设 box 是 cx,cy,w,h)
    x1 = int(max(0, bx - bw / 2))
    y1 = int(max(0, by - bh / 2))
    x2 = int(min(w if box_meta is None else 999999, bx + bw / 2))
    y2 = int(min(h if box_meta is None else 999999, by + bh / 2))

    if box_meta is not None:
        # 裁剪到原图边界
        x2 = min(x2, img_bgr.shape[1])
        y2 = min(y2, img_bgr.shape[0])

    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img_bgr, f"person:{score:.2f}", (x1, max(y1 - 5, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 画关键点
    points = []
    for i, (kx, ky, kconf) in enumerate(kpts):
        vis = kconf > 0.5
        if vis:
            px = int(kx * scale_x + offset_x)
            py = int(ky * scale_y + offset_y)
            points.append((px, py))
            cv2.circle(img_bgr, (px, py), 5, KPT_COLORS[i], -1)
            cv2.circle(img_bgr, (px, py), 5, (255, 255, 255), 1)
        else:
            points.append(None)
            px = int(kx * scale_x + offset_x)
            py = int(ky * scale_y + offset_y)
            cv2.drawMarker(img_bgr, (px, py), (128, 128, 128), cv2.MARKER_CROSS, 6, 1)

    # 画骨架
    for (i, j) in SKELETON:
        if points[i] and points[j]:
            cv2.line(img_bgr, points[i], points[j], (255, 128, 0), 2, cv2.LINE_AA)

    visible = sum(1 for p in points if p is not None)
    cv2.putText(img_bgr, f"Keypoints visible: {visible}/17", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    return img_bgr


def draw_batch(img_bgrs, results, input_size=640, box_metas=None, output_dir=".", prefix="result"):
    """批量绘制并保存"""
    os.makedirs(output_dir, exist_ok=True)
    saved_paths = []

    if box_metas is None:
        box_metas = [None] * len(img_bgrs)

    for idx, (img, res, meta) in enumerate(zip(img_bgrs, results, box_metas)):
        if img is None:
            continue
        drawn = draw_single(img, res, input_size, meta)
        out_name = f"{prefix}_{idx:04d}.jpg"
        out_path = os.path.join(output_dir, out_name)
        cv2.imwrite(out_path, drawn)
        saved_paths.append(out_path)
        status = "OK" if res else "No detection"
        print(f"  [{idx+1}/{len(img_bgrs)}] -> {out_path} ({status})")

    return saved_paths


def collect_images(args):
    """从命令行参数收集图片路径列表"""
    paths = []
    if args.image:
        paths.extend(args.image)
    if args.input_dir:
        ext = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        for f in sorted(Path(args.input_dir).glob("*")):
            if f.suffix.lower() in ext:
                paths.append(str(f))
    return paths


def parse_boxes_meta(meta_str):
    """解析 JSON 格式的检测框元数据"""
    if not meta_str:
        return None
    try:
        data = json.loads(meta_str)
        if isinstance(data, list):
            return data
        return [data]
    except Exception as e:
        print(f"解析 boxes-meta 失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="YOLOv8-pose Batch 推理")
    parser.add_argument("--model", required=True, help="ONNX 模型路径")
    parser.add_argument("--image", nargs="+", help="输入图片路径（可多张）")
    parser.add_argument("--input-dir", help="输入图片目录")
    parser.add_argument("--output-dir", default=".", help="输出目录")
    parser.add_argument("--output-prefix", default="pose", help="输出文件名前缀")
    parser.add_argument("--mode", default="decoded", choices=["raw","decoded","normalized"])
    parser.add_argument("--input-size", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--kpt-conf", type=float, default=0.5, help="关键点可见度阈值")
    # 两阶段专用参数
    parser.add_argument("--boxes-meta", help='检测框元数据 JSON，例: [{"x1":100,"y1":200,"x2":300,"y2":500}, ...]')
    parser.add_argument("--orig-size", help="原图尺寸，用于裁剪边界检查，格式: 1920x1080")
    args = parser.parse_args()

    # 收集图片
    image_paths = collect_images(args)
    if not image_paths:
        parser.error("请提供 --image 或 --input-dir")
    print(f"共加载 {len(image_paths)} 张图片")

    # 解析 boxes_meta（两阶段）
    box_metas = parse_boxes_meta(args.boxes_meta)
    if box_metas and len(box_metas) != len(image_paths):
        print(f"警告: boxes-meta 数量 ({len(box_metas)}) 与图片数量 ({len(image_paths)}) 不匹配，将忽略 meta")
        box_metas = None

    # 加载模型
    print(f"加载模型: {args.model}")
    sess = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    input_name = sess.get_inputs()[0].name
    print(f"  输入: {sess.get_inputs()[0].shape}, 输出: {[o.shape for o in sess.get_outputs()]}")

    # 批量预处理
    print("预处理...")
    batch_tensor, orig_sizes, img_bgrs = preprocess_batch(image_paths, args.input_size)
    print(f"  Batch tensor: {batch_tensor.shape}")

    # Batch 推理
    print("ONNX 推理...")
    outputs = sess.run(None, {input_name: batch_tensor})
    output = outputs[0]  # [B, N, 56]
    print(f"  输出形状: {output.shape}")

    # 批量后处理
    print("后处理...")
    results = decode_predictions_batch(
        output,
        mode=args.mode,
        input_size=args.input_size,
        conf_thres=args.conf,
        kpt_conf_thres=args.kpt_conf
    )
    detected = sum(1 for r in results if r is not None)
    print(f"  检测到人体: {detected}/{len(results)}")

    # 批量绘制
    print(f"保存结果到: {args.output_dir}")
    saved = draw_batch(
        img_bgrs, results,
        input_size=args.input_size,
        box_metas=box_metas,
        output_dir=args.output_dir,
        prefix=args.output_prefix
    )
    print(f"完成，共保存 {len(saved)} 张")


if __name__ == "__main__":
    main()