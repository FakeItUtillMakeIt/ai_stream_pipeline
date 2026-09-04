# HAL 能力矩阵

硬件抽象层（`src/hal`）按 **能力 → 厂商** 两层组织。本文档是各能力的**实现状态**权威来源：

- ✅ 已实现且已验证构建
- ⚠️ 已实现但依赖对应硬件，未真机验证
- ❌ 未实现（能力存在或由其他后端覆盖）
- 通过 `→` 引用的是覆盖该能力的实际实现

> NVDEC / FFmpeg 在 x86 + NVIDIA GPU 上即可真机验证（FFmpeg 需支持 CUDA hwaccel）；
> MPP / RGA / RKNN 需 RK3588 板子，DVPP 需昇腾设备。

| 能力 | nvidia | rk | ascend | cpu |
|---|---|---|---|---|
| 推理（通用） | TensorRT ✅ | RKNN ✅ | ❌ | OpenCV DNN ✅ |
| 检测 | TensorRT ✅ | RKNN ✅ | ❌ | ❌（节点 mock 兜底） |
| 姿态估计 | TensorRT ✅ | **RKNN ✅** | ❌ | ❌（节点 mock 兜底） |
| 动作识别 | TensorRT ✅ | RKNN ✅ | CANN ✅ | ❌ |
| 解码 | NVDEC ✅ | MPP ⚠️ | DVPP ⚠️ | FFmpeg ✅ |
| 编码 | → FFmpeg(nvenc) | MPP ⚠️ | ❌ | FFmpeg ✅ |
| 图像加速（预处理/绘制/NMS） | NPP+CUDA ✅ | RGA ⚠️ | DVPP ⚠️ | OpenCV ✅ |

## 注册与选择

每个能力由根目录工厂（`inference_engine_factory` / `detection_inference_engine_factory` /
`pose_estimation_factory` / `action_recognition_factory` / `video_codec_factory` /
`video_encoder_factory` / `image_accelerator_factory`）持有后端注册表：

- 后端在编译期按 `WITH_*` 选项编入，运行期通过静态注册宏注册到工厂
- `AUTO` 模式按固定优先级选择第一个 `isAvailable()` 为真的后端
  （TensorRT > RKNN > Ascend > CPU）
- 硬件后端（RKNN/NVDEC/MPP/RGA…）通过 **dlopen 惰性加载**：x86 编译主机
  无对应库时 `isAvailable()` 返回 false，自动回退下一优先级
- 编码器（`video_encoder_factory`）的 `AUTO` 在 RK 上选 `mpp_h264`，
  在 x86 上选 `ffmpeg_h264`（后者可通过 `encoder: h264_nvenc` 驱动 NVENC）

## 目录结构

```
src/hal/
├── infer/        {nvidia, rk, ascend, cpu}   推理引擎
├── encode/       {rk, cpu}                   编码（MPP / FFmpeg）
├── decode/       {nvidia, rk, ascend, cpu}   解码
├── image_accel/  {nvidia, rk, ascend, cpu}   图像加速
└── *_factory.cpp                             能力工厂

include/ai_stream/hal/                        抽象接口 + 共享工具
```

## 后端输出契约

各厂商引擎对同一能力输出**同语义张量**，节点层解耦：

- 检测：5 张量 `det_boxes[cxcywh] / det_scores / det_classes / det_batch_ids / det_num_dets`
- 姿态：`[num_persons, 8400, 56]`（4 box + 1 score + 51 kpt，输入 640 坐标系）
- 编码：全部 HAL 编码器输出 **AnnexB 包 + AnnexB extradata**，AVCDecoderConfigurationRecord
  由 muxer（容器）生成

## 待实现（roadmap）

| 项 | 说明 | 前置条件 |
|---|---|---|
| DVPP VENC（`encode/ascend/`） | Ascend 硬件编码器 | 昇腾设备 + CANN SDK |
| 独立 NVENC HAL（`encode/nvidia/`） | GPU 帧零拷贝编码 | CUDA + NvEncodeAPI，做全链路零拷贝时 |
| Ascend 检测/姿态/通用推理 | CANN 推理引擎补齐 | 昇腾设备 + CANN SDK |
| CPU 检测/姿态后端 | 补全 CPU 矩阵（当前节点 mock 兜底） | 低优先级 |