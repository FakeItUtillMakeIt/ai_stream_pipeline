#!/usr/bin/env python3
"""
VideoMAE微调脚本
用于在自定义数据集上微调VideoMAE模型进行动作识别

使用方法：
1. 准备数据集（参考README）
2. 激活环境：conda activate videomae_fune
3. 运行训练：python tools/train/videomae_finetune.py --data_dir /path/to/dataset --num_classes 6
"""

import os
import json
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# 设置多进程启动方式，避免 OpenCV 兼容性问题
mp.set_start_method('spawn', force=True)
from transformers import (
    VideoMAEForVideoClassification,
    VideoMAEImageProcessor,
    TrainingArguments,
    Trainer
)
import numpy as np
from PIL import Image
import cv2
from pathlib import Path


class VideoActionDataset(Dataset):
    """
    视频动作识别数据集
    
    数据集目录结构：
    dataset/
    ├── train/
    │   ├── action1/
    │   │   ├── video1.mp4
    │   │   └── video2.mp4
    │   └── action2/
    │       └── video3.mp4
    └── val/
        ├── action1/
        │   └── video4.mp4
        └── action2/
            └── video5.mp4
    """
    
    def __init__(self, root_dir, split='train', processor=None, 
                 num_frames=16, image_size=224):
        self.root_dir = root_dir
        self.split = split
        self.processor = processor
        self.num_frames = num_frames
        self.image_size = image_size
        
        # 获取类别
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        
        self.classes = sorted([d for d in os.listdir(split_dir) 
                              if os.path.isdir(os.path.join(split_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # 获取视频文件
        self.samples = []
        for cls_name in self.classes:
            cls_dir = os.path.join(split_dir, cls_name)
            for video_name in os.listdir(cls_dir):
                if video_name.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    video_path = os.path.join(cls_dir, video_name)
                    self.samples.append((video_path, self.class_to_idx[cls_name]))
        
        print(f"Loaded {len(self.samples)} videos for {split}")
        print(f"Classes: {self.classes}")
        print(f"Class to index: {self.class_to_idx}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        video_path, label = self.samples[idx]
        
        # 读取视频帧
        frames = self._load_video_frames(video_path)
        
        # 使用processor预处理
        if self.processor:
            inputs = self.processor(frames, return_tensors="pt")
            inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        else:
            # 手动预处理
            transform = transforms.Compose([
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            processed_frames = [transform(frame) for frame in frames]
            inputs = torch.stack(processed_frames, dim=1)  # [C, T, H, W]
            inputs = {'pixel_values': inputs}
        
        inputs['labels'] = torch.tensor(label, dtype=torch.long)
        
        return inputs
    
    def _load_video_frames(self, video_path):
        """从视频中均匀采样帧"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if total_frames == 0:
            # 如果无法读取视频，返回空白帧
            return [Image.fromarray(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8))] * self.num_frames
        
        # 均匀采样帧索引
        frame_indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                # 转换为RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
            else:
                if frames:
                    frames.append(frames[-1])
                else:
                    frames.append(Image.fromarray(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)))
        
        cap.release()
        
        # 确保帧数正确
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else Image.fromarray(np.zeros((self.image_size, self.image_size, 3), dtype=np.uint8)))
        
        return frames[:self.num_frames]


def compute_metrics(eval_pred):
    """计算评估指标"""
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


def main():
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE")
    parser.add_argument("--model_name", type=str, default="MCG-NJU/videomae-small",
                        help="预训练模型名称或路径")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据集目录")
    parser.add_argument("--output_dir", type=str, default="./models/videomae/finetuned",
                        help="输出目录")
    parser.add_argument("--num_classes", type=int, required=True,
                        help="动作类别数量")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="输入视频帧数")
    parser.add_argument("--image_size", type=int, default=224,
                        help="输入图像大小")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="训练轮数")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="批大小")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="学习率")
    parser.add_argument("--use_lora", action="store_true",
                        help="使用LoRA进行参数高效微调")
    parser.add_argument("--fp16", action="store_true",
                        help="使用FP16混合精度训练")
    parser.add_argument("--pretrained_weights", type=str, default=None,
                        help="预训练.pth权重路径（可选）")
    
    args = parser.parse_args()
    
    # 打印设备信息（多 GPU 兼容）
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPU(s)")
        for i in range(num_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("Using CPU")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载processor
    try:
        processor = VideoMAEImageProcessor.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Warning: Could not load processor from {args.model_name}: {e}")
        print("Using default processor...")
        processor = None
    
    # 加载模型
    model = VideoMAEForVideoClassification.from_pretrained(
        args.model_name,
        num_labels=args.num_classes,
        ignore_mismatched_sizes=True
    )
    
    # 如果指定了预训练权重，加载它
    if args.pretrained_weights:
        print(f"Loading pretrained weights from {args.pretrained_weights}")
        state_dict = torch.load(args.pretrained_weights, map_location='cpu', weights_only=False)
        if 'model' in state_dict:
            state_dict = state_dict['model']
        
        # 只加载encoder权重（忽略decoder和分类头）
        encoder_state = {k: v for k, v in state_dict.items() if not k.startswith('decoder.')}
        
        # 加载权重
        missing, unexpected = model.load_state_dict(encoder_state, strict=False)
        print(f"Loaded encoder weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        print(f"Missing keys (expected for new classification head): {missing}")
    
    # 如果使用LoRA
    if args.use_lora:
        from peft import LoraConfig, get_peft_model
        
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["query", "key", "value"],
            lora_dropout=0.1,
            bias="none",
            task_type="SEQ_CLS"
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    # 注意：不要手动 model.to(device)，让 Trainer 自动处理多 GPU
    
    # 创建数据集
    train_dataset = VideoActionDataset(
        args.data_dir, split='train',
        processor=processor,
        num_frames=args.num_frames,
        image_size=args.image_size
    )
    
    val_dataset = VideoActionDataset(
        args.data_dir, split='val',
        processor=processor,
        num_frames=args.num_frames,
        image_size=args.image_size
    )
    
    # 训练参数
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        fp16=args.fp16 and torch.cuda.is_available(),
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",  # 禁用wandb等
    )
    
    # 创建Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
    )
    
    # 开始训练
    print("Starting training...")
    trainer.train()
    
    # 保存模型
    print(f"Saving model to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    if processor:
        processor.save_pretrained(args.output_dir)
    
    # 保存类别映射
    class_mapping = {i: cls for i, cls in enumerate(train_dataset.classes)}
    with open(os.path.join(args.output_dir, "class_mapping.json"), "w") as f:
        json.dump(class_mapping, f, indent=2)
    
    print("Training completed!")
    print(f"Model saved to: {args.output_dir}")
    print(f"Class mapping: {class_mapping}")


if __name__ == "__main__":
    main()
