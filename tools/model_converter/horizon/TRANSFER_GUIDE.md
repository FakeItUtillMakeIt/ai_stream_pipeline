# Horizon BPU 模型转换指南

## 环境要求

| 项目 | 要求 |
|------|------|
| 平台 | x86_64 Ubuntu 22.04 |
| GPU | CUDA 12.6 + NVIDIA GPU（可选，加速量化） |
| Docker | 20.10.10+ |
| 工具链 | Horizon OpenExplorer Docker 镜像 |

## 快速开始

### 1. 在开发机上拉取 Docker 镜像

```bash
# S100P 专用镜像（CPU版，约 5GB）
docker pull registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0

# 或 GPU 版（更快，需要 NVIDIA 驱动）
docker pull registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_gpu:v3.7.0
```

### 2. 进入容器

```bash
docker run -it --rm \
    --network host \
    -v /path/to/project:/workspace \
    -w /workspace \
    registry.d-robotics.cc/deliver/ai_toolchain_ubuntu_22_s100_s600_cpu:v3.7.0 bash
```

### 3. 容器内转换

```bash
cd /workspace/tools/model_converter/horizon
./convert.sh /workspace/models/sevncevision/20260422_sevnce_14cls.onnx \
    --imgsz 224 \
    --march nash-m
```

### 4. 拷贝到板端

```bash
# 从开发机拷贝到板端
scp output/*.hbm root@<board_ip>:/home/sevnce/project/ai_stream_pipeline/models/sevncevision/
```

## 文件位置

```
ai_stream_pipeline/
├── models/sevncevision/
│   ├── 20260422_sevnce_14cls.onnx    # 原始 ONNX
│   └── 20260422_sevnce_14cls.hbm     # 转换后（在板端使用）
├── tools/model_converter/horizon/
│   ├── export_onnx.py                 # PyTorch → ONNX
│   ├── convert.sh                     # ONNX → HBM
│   ├── config_template.yaml           # 量化配置模板
│   └── TRANSFER_GUIDE.md              # 本文件
```

## 板端验证

```bash
# 在 RDK S100P 上验证模型
hrt_model_exec --model_file 20260422_sevnce_14cls.hbm \
    --func_name inference \
    --prefill --output output.json
```

## 常见问题

### Q: Docker 拉取超时
配置镜像加速：
```bash
sudo tee /etc/docker/daemon.json << 'EOF'
{
    "registry-mirrors": [
        "https://docker.1ms.run",
        "https://docker.xuanyuan.me"
    ]
}
EOF
sudo systemctl restart docker
```

### Q: 量化精度下降
- 确保 ONNX 导出时关闭 NMS
- 增加校准数据集数量
- 尝试 QAT 方案
