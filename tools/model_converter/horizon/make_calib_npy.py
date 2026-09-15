#!/usr/bin/env python3
# ============================================================================
# 把校准图片转换为 HBDK4 hb_compile 需要的 .npy 校准数据
#
# 新版 hb_compile 已弃用 cal_data_type，要求校准数据为 .npy。
# 预处理：BGR uint8 -> letterbox NxN (灰底 114) -> BGR转RGB -> HWC转NCHW -> float32
# 数值保持 [0,255] 原值；归一化交给 hb_compile 的 scale_value 处理
# （与官方 RDK 参考一致：校准数据 [0,255] + norm_type=data_scale, scale_value=1/255）
#
# 用法:
#   python3 make_calib_npy.py <图片目录> <输出npy目录> [尺寸,默认640]
# ============================================================================
import os
import sys
import numpy as np
import cv2


def letterbox_rgb(img_bgr: np.ndarray, size: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    scale = min(size / float(w), size / float(h))
    lw = max(1, int(round(w * scale)))
    lh = max(1, int(round(h * scale)))

    resized = cv2.resize(img_bgr, (lw, lh), interpolation=cv2.INTER_LINEAR)
    resized = resized.astype(np.float32)          # 保持 [0,255]

    canvas = np.full((size, size, 3), 114.0, dtype=np.float32)
    pad_x = (size - lw) // 2
    pad_y = (size - lh) // 2
    canvas[pad_y:pad_y + lh, pad_x:pad_x + lw] = resized

    rgb = canvas[..., ::-1]                      # BGR -> RGB
    chw = np.transpose(rgb, (2, 0, 1))           # HWC -> CHW
    return np.expand_dims(chw, 0).astype(np.float32)  # -> (1,3,H,W)


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 640

    os.makedirs(out_dir, exist_ok=True)
    exts = ('.jpg', '.jpeg', '.png', '.bmp')
    files = sorted(f for f in os.listdir(in_dir) if f.lower().endswith(exts))
    if not files:
        print(f"[ERROR] 目录内没有图片: {in_dir}")
        sys.exit(1)

    n = 0
    sample_shape = None
    for f in files:
        img = cv2.imread(os.path.join(in_dir, f))
        if img is None:
            print(f"[WARN] 读取失败，跳过: {f}")
            continue
        arr = letterbox_rgb(img, size)
        sample_shape = arr.shape
        out_name = os.path.splitext(f)[0] + ".npy"
        np.save(os.path.join(out_dir, out_name), arr)
        n += 1

    print(f"[INFO] 生成 {n} 个 npy -> {out_dir} (shape={sample_shape}, dtype=float32)")


if __name__ == "__main__":
    main()
