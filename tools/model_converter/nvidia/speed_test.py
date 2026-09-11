
import torch
import time
from ultralytics import YOLO

# 检查CUDA状态
print(f'CUDA可用: {torch.cuda.is_available()}')
print(f'CUDA版本: {torch.version.cuda}')
print(f'GPU设备: {torch.cuda.get_device_name(0)}')
print(f'当前设备: {torch.cuda.current_device()}')

# 加载模型
model = YOLO('../../models/sevncevision/20260422_sevnce_14cls.pt')

# 测试推理
img = '/home/sevnce/project/bus.jpg'

# CPU推理
model.to('cpu')
start = time.time()
results_cpu = model(img, verbose=False)
cpu_time = (time.time() - start) * 1000
print(f'CPU推理时间: {cpu_time:.1f}ms')

# GPU推理
model.to('cuda')
# 预热GPU
for _ in range(3):
    _ = model(img, verbose=False)
torch.cuda.synchronize()

start = time.time()
results_gpu = model(img, verbose=False)
torch.cuda.synchronize()
gpu_time = (time.time() - start) * 1000
print(f'GPU推理时间: {gpu_time:.1f}ms')
print(f'加速比: {cpu_time/gpu_time:.1f}x')
