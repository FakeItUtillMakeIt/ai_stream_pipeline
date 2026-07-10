#!/usr/bin/env python3
"""
VideoMAE 模型验证脚本
用于测试训练好的模型效果

使用方法：
python tools/train/validate.py --model_dir ./models/videomae/finetuned --data_dir /path/to/test_data
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from sklearn.metrics import classification_report, confusion_matrix
import json


def load_model(model_dir, device):
    """加载模型"""
    model = VideoMAEForVideoClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    # 加载处理器
    try:
        processor = VideoMAEImageProcessor.from_pretrained(model_dir)
    except:
        processor = None
    
    # 加载类别映射
    class_mapping_path = os.path.join(model_dir, "class_mapping.json")
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
            # 转换key为int
            class_mapping = {int(k): v for k, v in class_mapping.items()}
    else:
        class_mapping = {0: "climb", 1: "fight"}
    
    return model, processor, class_mapping


def load_video_frames(video_path, num_frames=16, image_size=224):
    """加载视频帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames == 0:
        return None
    
    # 均匀采样帧
    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))
        else:
            if frames:
                frames.append(frames[-1])
            else:
                frames.append(Image.fromarray(np.zeros((image_size, image_size, 3), dtype=np.uint8)))
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.fromarray(np.zeros((image_size, image_size, 3), dtype=np.uint8)))
    
    return frames[:num_frames]


def predict(model, processor, video_path, class_mapping, device, num_frames=16):
    """预测单个视频"""
    frames = load_video_frames(video_path, num_frames)
    if frames is None:
        return None, None, None
    
    # 预处理
    if processor:
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
        # 手动预处理
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        processed_frames = [transform(f) for f in frames]
        pixel_values = torch.stack(processed_frames, dim=1).unsqueeze(0).to(device)
        inputs = {'pixel_values': pixel_values}
    
    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # 获取预测结果
    probs = torch.softmax(logits, dim=-1)
    pred_idx = probs.argmax(-1).item()
    confidence = probs[0][pred_idx].item()
    
    pred_class = class_mapping.get(pred_idx, f"class_{pred_idx}")
    
    return pred_class, confidence, probs[0].cpu().numpy()


def validate(model_dir, data_dir, num_frames=16):
    """验证模型在测试集上的效果"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    model, processor, class_mapping = load_model(model_dir, device)
    print(f"Loaded model from {model_dir}")
    print(f"Classes: {class_mapping}")
    
    # 收集所有测试视频
    all_videos = []
    all_labels = []
    all_names = []
    
    for cls_name in os.listdir(data_dir):
        cls_dir = os.path.join(data_dir, cls_name)
        if not os.path.isdir(cls_dir):
            continue
        
        # 找到类别索引
        cls_idx = None
        for idx, name in class_mapping.items():
            if name == cls_name:
                cls_idx = idx
                break
        
        if cls_idx is None:
            print(f"Warning: Unknown class '{cls_name}', skipping")
            continue
        
        for video_name in os.listdir(cls_dir):
            if video_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(cls_dir, video_name)
                all_videos.append(video_path)
                all_labels.append(cls_idx)
                all_names.append(f"{cls_name}/{video_name}")
    
    print(f"\nFound {len(all_videos)} test videos")
    
    # 逐个预测
    correct = 0
    total = 0
    predictions = []
    true_labels = []
    
    for i, (video_path, true_label, video_name) in enumerate(zip(all_videos, all_labels, all_names)):
        pred_class, confidence, probs = predict(model, processor, video_path, class_mapping, device, num_frames)
        
        if pred_class is None:
            print(f"[{i+1}/{len(all_videos)}] {video_name}: Failed to load video")
            continue
        
        pred_idx = list(class_mapping.keys())[list(class_mapping.values()).index(pred_class)]
        
        is_correct = (pred_idx == true_label)
        if is_correct:
            correct += 1
        total += 1
        
        predictions.append(pred_idx)
        true_labels.append(true_label)
        
        status = "✓" if is_correct else "✗"
        print(f"[{i+1}/{len(all_videos)}] {video_name}: {pred_class} ({confidence:.2%}) {status}")
    
    # 打印统计
    print("\n" + "="*50)
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
    print("="*50)
    
    # 分类报告
    if total > 0:
        target_names = [class_mapping[i] for i in sorted(class_mapping.keys())]
        print("\nClassification Report:")
        print(classification_report(true_labels, predictions, target_names=target_names))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, predictions))
    
    return correct / total if total > 0 else 0


def main():
    parser = argparse.ArgumentParser(description="Validate VideoMAE model")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="训练好的模型目录")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="测试数据目录")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="输入视频帧数")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        return
    
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory not found: {args.data_dir}")
        return
    
    validate(args.model_dir, args.data_dir, args.num_frames)


if __name__ == "__main__":
    main()
