# HAL 能力矩阵

硬件抽象层（`src/hal`）按 **能力 → 厂商** 两层组织。本文档是各能力的**实现状态**权威来源：

- ✅ 已实现且已验证构建
- ⚠️ 已实现但依赖对应硬件，未真机验证
- ❌ 未实现（能力存在或由其他后端覆盖）
- 通过 `→` 引用的是覆盖该能力的实际实现

> NVDEC / FFmpeg 在 x86 + NVIDIA GPU 上即可真机验证（FFmpeg 需支持 CUDA hwaccel）；
> MPP / RGA / RKNN 需 RK3588 板子，DVPP 需昇腾设备；
> 地平线后端已在 RDK S100P 真机验证。

| 能力 | nvidia | rk | ascend | horizon | cpu |
|---|---|---|---|---|---|
| 推理（通用） | TensorRT ✅ | RKNN ✅ | ❌ | BPU ✅ | OpenCV DNN ✅ |
| 检测 | TensorRT ✅ | RKNN ✅ | ❌ | BPU ✅ | ❌（节点 mock 兜底） |
| 姿态估计 | TensorRT ✅ | **RKNN ✅** | ❌ | BPU ✅ | ❌（节点 mock 兜底） |
| 动作识别 | TensorRT ✅ | RKNN ✅ | CANN ✅ | BPU ✅ | ❌ |
| 解码 | NVDEC ✅ | MPP ⚠️ | DVPP ⚠️ | VPU(sp 硬解) ✅ | FFmpeg ✅ |
| 编码 | → FFmpeg(nvenc) | MPP ⚠️ | ❌ | VPU(sp 硬编) ✅ | FFmpeg ✅ |
| 图像加速（预处理/绘制/NMS） | NPP+CUDA ✅ | RGA ⚠️ | DVPP ⚠️ | OpenCV ✅ | OpenCV ✅ |

> 地平线说明：解码/编码经 `libspcdev.so` 的 `sp_*` API（VPU）；检测模型支持 **RGB int8** 与
> **NV12 双输入**两种；`resize_normalize` / OSD 绘制为 CPU OpenCV（VPU/GPU 硬件算子需 ION 零拷贝全链路，暂不采用）。

## 注册与选择

每个能力由根目录工厂（`inference_engine_factory` / `detection_inference_engine_factory` /
`pose_estimation_factory` / `action_recognition_factory` / `video_decoder_factory` /
`video_encoder_factory` / `image_accelerator_factory`）持有后端注册表：

- 后端在编译期按 `WITH_*` 选项编入，运行期通过静态注册宏注册到工厂
- `AUTO` 模式按固定优先级选择第一个 `isAvailable()` 为真的后端
  （TensorRT > RKNN > Ascend > 地平线 BPU > CPU）
- 硬件后端（RKNN/NVDEC/MPP/VPU…）通过 **dlopen 惰性加载**：x86 编译主机
  无对应库时 `isAvailable()` 返回 false，自动回退下一优先级
- 动态库加载统一使用 `include/ai_stream/hal/dl_library.h` 的 RAII `DlLibrary`（进程级共享、
  引用计数、线程安全，最后一个引用释放时 `dlclose`）；RKNN / MPP / RGA 后端已迁移，
  地平线 VPU/BPU 后端待板端 SDK 环境跟进
- 编码器（`video_encoder_factory`）的 `AUTO` 依次尝试 **`horizon_h264`（VPU）→
  `mpp_h264`（RK）→ `ffmpeg_h264`（软编/nvenc）**；sink/evidence 配置里写
  `"hw_encoder": true` 即自动选硬件（false 为软编），无需手写后端名；
  也可用 `"encoder": "h264_nvenc"` 等显式指定

## 目录结构

```
src/hal/
├── infer/        {nvidia, rk, ascend, horizon, cpu}   推理引擎
├── encode/       {rk, horizon, cpu}                   编码（MPP / VPU / FFmpeg）
├── decode/       {nvidia, rk, ascend, horizon, cpu}   解码
├── image_accel/  {nvidia, rk, ascend, horizon, cpu}   图像加速
└── *_factory.cpp                                       能力工厂

include/ai_stream/hal/                        抽象接口 + 共享工具
```

## 后端输出契约

各厂商引擎对同一能力输出**同语义张量**，节点层解耦：

- 检测：5 张量 `det_boxes[cxcywh] / det_scores / det_classes / det_batch_ids / det_num_dets`
- 姿态：`[num_persons, 8400, 56]`（4 box + 1 score + 51 kpt，输入 640 坐标系）
- 编码：全部 HAL 编码器输出 **AnnexB 包 + AnnexB extradata**，AVCDecoderConfigurationRecord
  由 muxer（容器）生成

## 地平线（RDK S100P）加速现状

| 环节 | 实现 | 说明 |
|---|---|---|
| 解码 | ✅ VPU 硬解 | `libspcdev.so` 的 `sp_*`，输出 NV12 |
| 编码 | ✅ VPU 硬编 | 同上；管线写 `"hw_encoder": true` 自动选 |
| 推理 | ✅ BPU | 检测模型支持 **RGB int8** 与 **NV12 双输入** |
| 预处理（resize/归一化） | ⚠️ CPU（OpenCV） | VPU 算子需 ION 物理内存，暂不采用 |
| OSD 绘制 | ⚠️ CPU（OpenCV） | 无"画到内存"的 2D 硬件；SP 显示 OSD 仅适用显示输出 |
| NMS | CPU | — |

**NV12 直通路径**：`ffmpeg_decode` 的 `output_nv12` + `detection_infer` 的 `input_nv12`
+ 引擎 NV12 双输入（自动解析动态 stride），跳过 `resize_normalize` 的 float 预处理，
直接喂解码得到的 NV12（管线上 `decode → infer` 直连，`decode` 设 `output_nv12`、`infer` 设 `input_nv12`）。

## 待实现（roadmap）

| 项 | 说明 | 前置条件 |
|---|---|---|
| 地平线硬件 resize（VPU VPS / `hbVPResize`） | VPS 仅支持等比下采样（如 1280×720→640×360），且需 `sp_module_bind(decoder→vps)` 拉流架构 + ION 物理内存；当前为 CPU letterbox | 解码侧改 SP 拉流 + ION 零拷贝 |
| 地平线硬件绘制（OSD） | 仅有 `sp_display_draw_rect`（画到 SP 显示层）；RTMP/evidence 输出无硬件绘制路径 | 无（改用 GPU 需上传/下载，不划算） |
| DVPP VENC（`encode/ascend/`） | Ascend 硬件编码器 | 昇腾设备 + CANN SDK |
| 独立 NVENC HAL（`encode/nvidia/`） | 当前 NVENC 经 FFmpeg（`h264_nvenc`）实现；独立直驱 NvEncodeAPI 用于 GPU 帧零拷贝输入。**暂缓**：sink 编码输入为 OSD 输出的 host BGR，零拷贝输入需 OSD 输出留显存（卡在 GPU 文字渲染） | NVIDIA Video Codec SDK 头文件 + OSD GPU 化 |
| Ascend 检测/姿态/通用推理 | CANN 推理引擎补齐 | 昇腾设备 + CANN SDK |
| CPU 检测/姿态后端 | 补全 CPU 矩阵（当前节点 mock 兜底） | 低优先级 |