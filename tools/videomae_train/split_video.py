#!/usr/bin/env python3
"""
视频切分脚本
将长视频切分为短视频片段用于训练

使用方法：
python tools/train/split_video.py --input /path/to/long_video.mp4 --output_dir /path/to/output --duration 5
"""

import os
import argparse
import cv2
import numpy as np
from pathlib import Path


def split_video(input_path, output_dir, duration=5, overlap=0.5, min_duration=2):
    """
    切分视频
    
    Args:
        input_path: 输入视频路径
        output_dir: 输出目录
        duration: 每个片段时长（秒）
        overlap: 重叠比例（0-1）
        min_duration: 最小片段时长（秒）
    """
    # 打开视频
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total_duration = total_frames / fps
    
    print(f"Input video: {input_path}")
    print(f"FPS: {fps}, Duration: {total_duration:.2f}s, Total frames: {total_frames}")
    
    # 计算切分参数
    frame_duration = int(duration * fps)
    step = int(frame_duration * (1 - overlap))
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取视频文件名（不含扩展名）
    video_name = Path(input_path).stem
    
    # 切分视频
    clip_idx = 0
    start_frame = 0
    
    while start_frame + frame_duration <= total_frames:
        end_frame = start_frame + frame_duration
        
        # 设置视频位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # 输出视频路径
        output_path = os.path.join(output_dir, f"{video_name}_clip{clip_idx:04d}.mp4")
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (int(cap.get(3)), int(cap.get(4))))
        
        # 写入帧
        for _ in range(frame_duration):
            ret, frame = cap.read()
            if ret:
                out.write(frame)
            else:
                break
        
        out.release()
        
        print(f"Saved clip {clip_idx}: frames {start_frame}-{end_frame} ({start_frame/fps:.2f}s - {end_frame/fps:.2f}s)")
        
        clip_idx += 1
        start_frame += step
    
    cap.release()
    print(f"\nTotal clips saved: {clip_idx}")
    print(f"Output directory: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Split long video into clips")
    parser.add_argument("--input", type=str, required=True,
                        help="输入视频路径")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出目录")
    parser.add_argument("--duration", type=float, default=5,
                        help="每个片段时长（秒），默认5秒")
    parser.add_argument("--overlap", type=float, default=0.5,
                        help="重叠比例（0-1），默认0.5")
    parser.add_argument("--min_duration", type=float, default=2,
                        help="最小片段时长（秒），默认2秒")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        return
    
    split_video(args.input, args.output_dir, args.duration, args.overlap, args.min_duration)


if __name__ == "__main__":
    main()
