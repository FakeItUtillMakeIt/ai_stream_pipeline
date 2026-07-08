#!/usr/bin/env python3
"""
VideoMAE 单视频推理脚本
支持：
1. 单个视频分类
2. 未知类别检测（置信度阈值）

使用方法：
python tools/train/predict.py --video /path/to/video.mp4 --model_dir ./models/videomae/finetuned
"""

import os
import argparse
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor


def load_model(model_dir, device):
    """加载模型"""
    model = VideoMAEForVideoClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()
    
    try:
        processor = VideoMAEImageProcessor.from_pretrained(model_dir)
    except:
        processor = None
    
    # 加载类别映射
    import json
    class_mapping_path = os.path.join(model_dir, "class_mapping.json")
    if os.path.exists(class_mapping_path):
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
            class_mapping = {int(k): v for k, v in class_mapping.items()}
    else:
        class_mapping = {0: "climb", 1: "fight"}
    
    return model, processor, class_mapping


def load_video_frames(video_path, num_frames=16):
    """加载视频帧"""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames == 0:
        print(f"Error: Cannot read video {video_path}")
        return None, None
    
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
                frames.append(Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
    
    cap.release()
    
    while len(frames) < num_frames:
        frames.append(frames[-1] if frames else Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8)))
    
    return frames[:num_frames], fps


def predict(video_path, model, processor, class_mapping, device, 
            num_frames=16, threshold=0.7):
    """
    预测单个视频
    
    Args:
        video_path: 视频路径
        model: 模型
        processor: 图像处理器
        class_mapping: 类别映射
        device: 设备
        num_frames: 采样帧数
        threshold: 置信度阈值（低于此值判定为未知）
    
    Returns:
        预测类别、置信度、所有类别概率
    """
    frames, fps = load_video_frames(video_path, num_frames)
    if frames is None:
        return None, None, None, None
    
    # 预处理
    if processor:
        inputs = processor(frames, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
    else:
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
    
    # 判断是否为未知类别
    if confidence < threshold:
        pred_class = "unknown"
    else:
        pred_class = class_mapping.get(pred_idx, f"class_{pred_idx}")
    
    # 所有类别概率
    all_probs = {class_mapping.get(i, f"class_{i}"): probs[0][i].item() 
                 for i in range(len(class_mapping))}
    
    return pred_class, confidence, all_probs, fps


def main():
    parser = argparse.ArgumentParser(description="Predict video action")
    parser.add_argument("--video", type=str, required=True,
                        help="视频文件路径")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="模型目录")
    parser.add_argument("--threshold", type=float, default=0.7,
                        help="置信度阈值（低于此值判定为未知，默认0.7）")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="采样帧数")
    
    args = parser.parse_args()
    
    # 检查文件
    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        return
    
    if not os.path.exists(args.model_dir):
        print(f"Error: Model not found: {args.model_dir}")
        return
    
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from {args.model_dir}...")
    model, processor, class_mapping = load_model(args.model_dir, device)
    print(f"Classes: {class_mapping}")
    print(f"Threshold: {args.threshold}")
    print("="*50)
    
    # 预测
    print(f"Video: {args.video}")
    pred_class, confidence, all_probs, fps = predict(
        args.video, model, processor, class_mapping, device,
        args.num_frames, args.threshold
    )
    
    if pred_class is None:
        print("Error: Failed to process video")
        return
    
    # 打印结果
    print(f"\nResult: {pred_class}")
    print(f"Confidence: {confidence:.2%}")
    print(f"\nAll probabilities:")
    for cls, prob in sorted(all_probs.items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"  {cls:10s}: {prob:.2%} {bar}")
    
    # 返回结果（可用于脚本调用）
    return {
        "class": pred_class,
        "confidence": confidence,
        "probabilities": all_probs
    }


if __name__ == "__main__":
    main()
