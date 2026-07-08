#!/usr/bin/env python3
"""
数据集准备脚本
用于将原始视频数据转换为训练所需的目录结构

使用方法：
python tools/train/prepare_dataset.py --input_dir /path/to/raw_videos --output_dir /path/to/dataset
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
import json


def scan_videos(input_dir, extensions=('.mp4', '.avi', '.mov', '.mkv')):
    """扫描目录中的视频文件"""
    video_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(extensions):
                video_files.append(os.path.join(root, file))
    return video_files


def organize_by_class(video_files):
    """按类别组织视频文件"""
    class_videos = defaultdict(list)
    
    for video_path in video_files:
        # 尝试从路径中提取类别
        # 假设目录结构是：input_dir/class_name/video.mp4
        path_parts = Path(video_path).parts
        if len(path_parts) >= 2:
            # 取倒数第二个目录作为类别名
            class_name = path_parts[-2]
        else:
            class_name = "unknown"
        
        class_videos[class_name].append(video_path)
    
    return dict(class_videos)


def split_dataset(class_videos, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """划分训练集、验证集和测试集"""
    random.seed(seed)
    
    splits = {'train': {}, 'val': {}, 'test': {}}
    
    for class_name, videos in class_videos.items():
        random.shuffle(videos)
        
        n = len(videos)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        
        splits['train'][class_name] = videos[:n_train]
        splits['val'][class_name] = videos[n_train:n_train + n_val]
        splits['test'][class_name] = videos[n_train + n_val:]
    
    return splits


def create_dataset_structure(splits, output_dir):
    """创建数据集目录结构并复制文件"""
    for split_name, class_videos in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        for class_name, videos in class_videos.items():
            class_dir = os.path.join(split_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)
            
            for video_path in videos:
                # 保持原始文件名，但添加视频源信息
                video_name = os.path.basename(video_path)
                dest_path = os.path.join(class_dir, video_name)
                
                # 复制文件
                shutil.copy2(video_path, dest_path)
                print(f"Copied: {video_path} -> {dest_path}")
    
    return splits


def main():
    parser = argparse.ArgumentParser(description="Prepare video dataset for training")
    parser.add_argument("--input_dir", type=str, required=True,
                        help="输入视频目录")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="输出数据集目录")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.1,
                        help="验证集比例")
    parser.add_argument("--test_ratio", type=float, default=0.1,
                        help="测试集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--dry_run", action="store_true",
                        help="只显示统计信息，不实际复制文件")
    
    args = parser.parse_args()
    
    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        return
    
    # 扫描视频
    print(f"Scanning videos in {args.input_dir}...")
    video_files = scan_videos(args.input_dir)
    print(f"Found {len(video_files)} videos")
    
    if len(video_files) == 0:
        print("No videos found!")
        return
    
    # 按类别组织
    class_videos = organize_by_class(video_files)
    print(f"\nFound {len(class_videos)} classes:")
    for class_name, videos in class_videos.items():
        print(f"  {class_name}: {len(videos)} videos")
    
    # 划分数据集
    splits = split_dataset(class_videos, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    
    # 显示统计信息
    print("\nDataset split:")
    for split_name, class_videos in splits.items():
        total = sum(len(videos) for videos in class_videos.values())
        print(f"  {split_name}: {total} videos")
        for class_name, videos in class_videos.items():
            print(f"    {class_name}: {len(videos)} videos")
    
    if args.dry_run:
        print("\nDry run mode - no files copied")
        return
    
    # 创建数据集结构
    print(f"\nCreating dataset structure in {args.output_dir}...")
    create_dataset_structure(splits, args.output_dir)
    
    # 保存数据集信息
    dataset_info = {
        "classes": list(class_videos.keys()),
        "num_classes": len(class_videos),
        "splits": {
            split_name: {
                "total": sum(len(v) for v in class_videos.values()),
                "classes": {cls: len(videos) for cls, videos in class_videos.items()}
            }
            for split_name, class_videos in splits.items()
        }
    }
    
    info_path = os.path.join(args.output_dir, "dataset_info.json")
    with open(info_path, "w") as f:
        json.dump(dataset_info, f, indent=2)
    
    print(f"\nDataset prepared successfully!")
    print(f"Dataset info saved to: {info_path}")
    
    # 生成使用说明
    print(f"\nUsage:")
    print(f"1. 激活环境: conda activate videomae_fune")
    print(f"2. 开始训练:")
    print(f"   python tools/train/videomae_finetune.py \\")
    print(f"     --data_dir {args.output_dir} \\")
    print(f"     --num_classes {len(class_videos)} \\")
    print(f"     --output_dir ./models/videomae/finetuned \\")
    print(f"     --num_epochs 50 \\")
    print(f"     --batch_size 8 \\")
    print(f"     --fp16")


if __name__ == "__main__":
    main()
