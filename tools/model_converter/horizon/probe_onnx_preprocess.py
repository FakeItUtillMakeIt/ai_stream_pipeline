#!/usr/bin/env python3
# ============================================================================
# 探测 ONNX 检测模型正确的输入预处理
#
# 在 x86(有 onnxruntime) 上运行，用一张包含目标的真实图，穷举常见预处理组合，
# 打印每种组合的 cls_max 与 NMS 后框数。正确的那组应当是：cls_max 较高、
# NMS 后框数少而合理（而不是 0 或几百）。
#
# 依赖: pip install onnxruntime opencv-python numpy
# 用法:
#   python3 probe_onnx_preprocess.py --onnx model.onnx --image frame.jpg [--size 640]
# ============================================================================
import argparse
import numpy as np
import cv2
import onnxruntime as ort


def letterbox(img, size, pad):
    h, w = img.shape[:2]
    s = min(size / w, size / h)
    lw, lh = max(1, round(w * s)), max(1, round(h * s))
    r = cv2.resize(img, (lw, lh), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    c = np.full((size, size, 3), float(pad), np.float32)
    px, py = (size - lw) // 2, (size - lh) // 2
    c[py:py + lh, px:px + lw] = r
    return c


def stretch(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR).astype(np.float32)


def nms_keep(o, th=0.25, iou=0.45):
    cx, cy, bw, bh = o[0], o[1], o[2], o[3]
    mx = o[4:].max(0)
    cl = o[4:].argmax(0)
    idx = list(np.where(mx > th)[0])
    idx.sort(key=lambda i: -mx[i])
    keep = []
    for i in idx:
        b = (cx[i], cy[i], bw[i], bh[i], cl[i])
        ok = True
        for k in keep:
            if k[4] != b[4]:
                continue
            x1 = max(b[0] - b[2] / 2, k[0] - k[2] / 2); y1 = max(b[1] - b[3] / 2, k[1] - k[3] / 2)
            x2 = min(b[0] + b[2] / 2, k[0] + k[2] / 2); y2 = min(b[1] + b[3] / 2, k[1] + k[3] / 2)
            it = max(0, x2 - x1) * max(0, y2 - y1)
            if it / (b[2] * b[3] + k[2] * k[3] - it + 1e-6) > iou:
                ok = False; break
        if ok:
            keep.append(b)
    return float(mx.max()), len(idx), len(keep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--onnx', required=True)
    ap.add_argument('--image', required=True)
    ap.add_argument('--size', type=int, default=640)
    a = ap.parse_args()

    sess = ort.InferenceSession(a.onnx, providers=['CPUExecutionProvider'])
    inp = sess.get_inputs()[0]
    print("input:", inp.name, inp.shape, inp.type)
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit("read image failed")

    size = a.size
    combos = []
    for ch in ('rgb', 'bgr'):
        for mode in ('letterbox114', 'stretch'):
            for rng in ('01', '255', 'pm1', 'imagenet'):
                combos.append((ch, mode, rng))

    print("%-6s %-12s %-9s %8s %8s %6s" % ("chan", "resize", "range", "cls_max", "cand", "keep"))
    for ch, mode, rng in combos:
        c = letterbox(img, size, 114) if mode == 'letterbox114' else stretch(img, size)
        if ch == 'rgb':
            c = c[..., ::-1]
        if rng == '01':
            c = c / 255.0
        elif rng == 'pm1':
            c = c / 127.5 - 1.0
        elif rng == 'imagenet':
            c = (c / 255.0 - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        # rng == '255' 保持不变
        x = np.transpose(c, (2, 0, 1))[None].astype(np.float32)
        o = sess.run(None, {inp.name: x})[0][0]
        if o.shape[0] < 5:  # 输出不是 [4+nc, N]
            print("%-6s %-12s %-9s  (output shape %s)" % (ch, mode, rng, o.shape)); continue
        mm, cand, keep = nms_keep(o)
        print("%-6s %-12s %-9s %8.3f %8d %6d" % (ch, mode, rng, mm, cand, keep))


if __name__ == '__main__':
    main()
