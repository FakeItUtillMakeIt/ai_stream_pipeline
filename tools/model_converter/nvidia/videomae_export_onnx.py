#!/usr/bin/env python3
"""
VideoMAE模型转换脚本
将VideoMAE模型从PyTorch转换为ONNX，再转换为TensorRT引擎

使用方法：
1. 安装依赖：pip install transformers torch onnx onnxruntime
2. 导出ONNX：python videomae_export_onnx.py --output videomae_small.onnx
3. 转换TensorRT：./videomae_onnx2trt.sh videomae_small.onnx
"""

import argparse
import torch
import torch.onnx
from transformers import VideoMAEForVideoClassification


def export_videomae_to_onnx(
    model_name: str = "MCG-NJU/videomae-small-finetuned-ssv2",
    output_path: str = "videomae_small.onnx",
    num_frames: int = 16,
    image_size: int = 224,
    opset_version: int = 14
):
    """
    导出VideoMAE模型为ONNX格式
    
    Args:
        model_name: HuggingFace模型名称或本地模型路径
        output_path: 输出ONNX文件路径
        num_frames: 输入视频帧数
        image_size: 输入图像尺寸
        opset_version: ONNX opset版本
    """
    print(f"Loading model: {model_name}")
    
    # 加载模型
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    model.eval()
    
    # 创建dummy输入
    # VideoMAE输入格式: [batch_size, num_frames, channels, height, width]
    dummy_input = torch.randn(1, num_frames, 3, image_size, image_size)
    
    print(f"Exporting to ONNX: {output_path}")
    
    # 导出ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"Model exported successfully to {output_path}")
    
    # 验证模型
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(output_path)
        print(f"ONNX model validated successfully")
        print(f"Input name: {session.get_inputs()[0].name}")
        print(f"Input shape: {session.get_inputs()[0].shape}")
        print(f"Output name: {session.get_outputs()[0].name}")
        print(f"Output shape: {session.get_outputs()[0].shape}")
    except Exception as e:
        print(f"Warning: Could not validate ONNX model: {e}")


def main():
    parser = argparse.ArgumentParser(description="Export VideoMAE to ONNX")
    parser.add_argument("--model", type=str, default="MCG-NJU/videomae-small-finetuned-ssv2",
                        help="HuggingFace model name or local path")
    parser.add_argument("--output", type=str, default="videomae_small.onnx",
                        help="Output ONNX file path")
    parser.add_argument("--num-frames", type=int, default=16,
                        help="Number of input frames")
    parser.add_argument("--image-size", type=int, default=224,
                        help="Input image size")
    parser.add_argument("--opset", type=int, default=14,
                        help="ONNX opset version")
    
    args = parser.parse_args()
    
    export_videomae_to_onnx(
        model_name=args.model,
        output_path=args.output,
        num_frames=args.num_frames,
        image_size=args.image_size,
        opset_version=args.opset
    )


if __name__ == "__main__":
    main()
