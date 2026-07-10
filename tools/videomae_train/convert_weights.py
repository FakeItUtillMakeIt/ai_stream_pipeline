#!/usr/bin/env python3
"""
将VideoMAE原始.pth权重转换为HuggingFace格式
用于加载预训练权重进行微调
"""

import torch
import os
import json
from collections import OrderedDict


def convert_videomae_weights(pth_path, output_dir):
    """
    转换VideoMAE .pth权重为HuggingFace格式
    
    Args:
        pth_path: 原始.pth权重路径
        output_dir: 输出目录
    """
    # 加载权重
    state_dict = torch.load(pth_path, map_location='cpu', weights_only=False)
    model_state = state_dict['model']
    
    # 完整的key映射表
    key_mapping = {
        'encoder.patch_embed.proj.weight': 'videomae.embeddings.patch_embeddings.projection.weight',
        'encoder.patch_embed.proj.bias': 'videomae.embeddings.patch_embeddings.projection.bias',
        'encoder.norm.weight': 'fc_norm.weight',
        'encoder.norm.bias': 'fc_norm.bias',
    }
    
    # 逐层映射
    layer_mapping = {
        'norm1.weight': 'layernorm_before.weight',
        'norm1.bias': 'layernorm_before.bias',
        'attn.proj.weight': 'attention.output.dense.weight',
        'attn.proj.bias': 'attention.output.dense.bias',
        'norm2.weight': 'layernorm_after.weight',
        'norm2.bias': 'layernorm_after.bias',
        'mlp.fc1.weight': 'intermediate.dense.weight',
        'mlp.fc1.bias': 'intermediate.dense.bias',
        'mlp.fc2.weight': 'output.dense.weight',
        'mlp.fc2.bias': 'output.dense.bias',
    }
    
    encoder_state = OrderedDict()
    for k, v in model_state.items():
        if k.startswith('encoder.'):
            # 检查是否在直接映射表中
            if k in key_mapping:
                new_key = key_mapping[k]
                encoder_state[new_key] = v
            # 检查是否是block层
            elif k.startswith('encoder.blocks.'):
                # 提取层号和剩余部分
                parts = k.split('.')
                layer_num = parts[2]  # blocks.X
                remaining = '.'.join(parts[3:])  # norm1.weight等
                
                # 处理qkv权重拆分
                if remaining == 'attn.qkv.weight':
                    # qkv.weight [3*hidden, hidden] -> query.weight, key.weight, value.weight
                    hidden_size = v.shape[1]
                    q_weight = v[:hidden_size, :]
                    k_weight = v[hidden_size:2*hidden_size, :]
                    v_weight = v[2*hidden_size:, :]
                    
                    encoder_state[f'videomae.encoder.layer.{layer_num}.attention.attention.query.weight'] = q_weight
                    encoder_state[f'videomae.encoder.layer.{layer_num}.attention.attention.key.weight'] = k_weight
                    encoder_state[f'videomae.encoder.layer.{layer_num}.attention.attention.value.weight'] = v_weight
                elif remaining == 'attn.q_bias':
                    # q_bias -> query.bias, key.bias (key没有bias)
                    encoder_state[f'videomae.encoder.layer.{layer_num}.attention.attention.query.bias'] = v
                    # key没有bias，跳过
                elif remaining == 'attn.v_bias':
                    # v_bias -> value.bias
                    encoder_state[f'videomae.encoder.layer.{layer_num}.attention.attention.value.bias'] = v
                elif remaining in layer_mapping:
                    new_key = f'videomae.encoder.layer.{layer_num}.{layer_mapping[remaining]}'
                    encoder_state[new_key] = v
    
    print(f"Original model has {len(model_state)} parameters")
    print(f"Encoder has {len(encoder_state)} parameters")
    print(f"Sample keys: {list(encoder_state.keys())[:5]}")
    
    # 保存encoder权重
    os.makedirs(output_dir, exist_ok=True)
    encoder_path = os.path.join(output_dir, 'pytorch_model.bin')
    torch.save(encoder_state, encoder_path)
    print(f"Saved encoder weights to {encoder_path}")
    
    return encoder_state


def create_config_from_weights(encoder_state, num_labels=6):
    """根据权重创建config.json"""
    # 从权重中推断模型配置
    hidden_size = encoder_state['fc_norm.weight'].shape[0]  # 384 for small, 768 for base
    
    config = {
        "architectures": ["VideoMAEForVideoClassification"],
        "image_size": 224,
        "num_channels": 3,
        "patch_size": 16,
        "num_frames": 16,
        "hidden_size": hidden_size,
        "num_hidden_layers": 12,
        "num_attention_heads": hidden_size // 64,
        "intermediate_size": hidden_size * 4,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "attention_probs_dropout_prob": 0.0,
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-12,
        "num_labels": num_labels,
        "model_type": "videomae"
    }
    
    return config


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth_path", type=str, default="./models/videomae/videomae_small.pth")
    parser.add_argument("--output_dir", type=str, default="./models/videomae/pretrained")
    parser.add_argument("--num_labels", type=int, default=6)
    args = parser.parse_args()
    
    # 转换权重
    encoder_state = convert_videomae_weights(args.pth_path, args.output_dir)
    
    # 创建config
    config = create_config_from_weights(encoder_state, args.num_labels)
    config_path = os.path.join(args.output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")
    
    # 创建preprocessor_config
    preprocessor_config = {
        "do_center_crop": True,
        "do_normalize": True,
        "do_rescale": True,
        "do_resize": True,
        "image_mean": [0.485, 0.456, 0.406],
        "image_processor_type": "VideoMAEImageProcessor",
        "image_std": [0.229, 0.224, 0.225],
        "resample": "bicubic",
        "rescale_factor": 0.00392156862745098,
        "size": {"shortest_edge": 224}
    }
    preprocessor_path = os.path.join(args.output_dir, 'preprocessor_config.json')
    with open(preprocessor_path, 'w') as f:
        json.dump(preprocessor_config, f, indent=2)
    print(f"Saved preprocessor config to {preprocessor_path}")
    
    print("\nConversion complete!")
    print(f"Files saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
