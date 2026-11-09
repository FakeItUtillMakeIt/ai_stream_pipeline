# HAL 能力矩阵

硬件抽象层（`src/hal`）按 **能力 → 厂商** 两层组织。本文档是各能力的**实现状态**权威来源：

- ✅ 已实现且已验证构建
- ⚠️ 已实现但依赖对应硬件，未真机验证
- ❌ 未实现（能力存在或由其他后端覆盖）
- 通过 `→` 引用的是覆盖该能力的实际实现

> NVDEC / FFmpeg 在 x86 + NVIDIA GPU 上即可真机验证（FFmpeg 需支持 CUDA hwaccel）；
> MPP / RGA / RKNN 需 RK3588 板子，DVPP 需昇腾设备；
> 地平线后端已在 RDK S100P 真机验证。

| 能力 | nvidia | intel | rk | ascend | horizon | cpu |
|---|---|---|---|---|---|---|
| 推理（通用） | TensorRT ✅ | ❌（OpenVINO 仅实现检测） | RKNN ✅ | ❌ | BPU ✅ | OpenCV DNN ✅ |
| 检测 | TensorRT ✅ | OpenVINO ✅（CPU 已验证；iGPU/NPU 需 compute runtime） | RKNN ✅ | **ACL ✅**（`.om`，310P3 已验证） | BPU ✅ | ❌（节点 mock 兜底） |
| 姿态估计 | TensorRT ✅ | ❌ | **RKNN ✅** | ❌ | BPU ✅ | ❌（节点 mock 兜底） |
| 动作识别 | TensorRT ✅ | ❌ | RKNN ✅ | **ACL ✅**（`.om`，310P3 已验证） | BPU ✅ | ❌ |
| 解码 | NVDEC ✅ | VAAPI ✅（需 `/dev/dri/renderD*` 权限） | MPP ⚠️ | **DVPP ✅**（H264/H265，310P3 已验证） | VPU(sp 硬解) ✅ | FFmpeg ✅ |
| 编码 | → FFmpeg(nvenc) | VAAPI ✅（h264_vaapi） | MPP ⚠️ | **DVPP VENC ✅**（H.264，310P3 已验证） | VPU(sp 硬编) ✅ | FFmpeg ✅ |
| 图像加速（预处理/绘制/NMS） | NPP+CUDA ✅ | OpenCV（CPU） | RGA ⚠️ | **DVPP VPC resize ✅**（NMS/绘制 CPU） | OpenCV ✅ | OpenCV ✅ |

> Intel 说明：编解码走 FFmpeg 的 VAAPI hwaccel（`h264_vaapi` / `AV_PIX_FMT_VAAPI`），
> 不链接 `libva`，运行时要求内核 i915/xe + `intel-media-va-driver`（iHD）且服务用户属于
> `render` 组；不满足时 `isAvailable()` 返回 false，自动回退 FFmpeg 软解/软编。
> 推理走 OpenVINO C API（`libopenvino_c`），AUTO 优先 `GPU.0/GPU`，不可用时回退 `CPU`；
> iGPU/NPU 推理还需安装 Intel compute runtime（`intel-opencl-icd` / Level Zero）。
> OpenVINO 当前只实现检测 5 张量契约，姿态/动作仍走 TensorRT/RKNN/BPU。
> 检测节点用 `"detector_config": {"backend": "openvino", "model_path": "xxx.onnx"}`；
> 不写 `backend` 时 AUTO 会按扩展名选后端（`.engine`→TensorRT，`.onnx/.xml`→OpenVINO）。

> 地平线说明：解码/编码经 `libspcdev.so` 的 `sp_*` API（VPU）；检测模型支持 **RGB int8** 与
> **NV12 双输入**两种；`resize_normalize` / OSD 绘制为 CPU OpenCV（VPU/GPU 硬件算子需 ION 零拷贝全链路，暂不采用）。

> 昇腾说明：`cmake/FindAscend.cmake` 已适配 CANN 8.x 的
> `<root>/x86_64-linux/{include,lib64}` 布局，`WITH_ASCEND=ON` 可找到并链接
> `libascendcl` / `libacl_dvpp`。当前已实现：
> **ACL 检测推理**（`.om`，ATC 转换；输出 5 张量契约）、**ACL 动作识别**
> （VideoMAE `.om`，softmax/argmax）、**DVPP VDEC 硬解**（H264/H265；AVCC 自动转 AnnexB）、
> **DVPP VPC resize**（BGR888 进/出，归一化在 CPU）与 **DVPP VENC H.264 硬编**
> （open 时编码一帧灰图提取 SPS/PPS，首个真实帧强制 IDR）。运行前置：服务用户在
> `HwHiAiUser` 组（可访问 `/dev/davinci*`）。构建产物已通过 RPATH 指向 CANN
> lib64/driver 目录，通常无需手动 `source set_env.sh`；CANN 安装位置变化时再 source。
> DVPP 没有 NMS/绘制算子，NMS 与 OSD 绘制仍走 CPU；VENC 目前只支持 H.264。
> 检测节点配置 `"backend": "ascend"` + `.om`，AUTO 也会按 `.om` 扩展名选择 ACL 后端；
> 动作识别节点按 `.om` 自动选 Ascend；sink/evidence 写 `"hw_encoder": true` 时 AUTO
> 会选 `dvpp_h264`。未显式设置 `output_width/output_height` 时，sink 用首帧输入宽高。
> 注意：DVPP VDEC 使用回调 + `aclrtProcessReport`，节点当前仍是“一包一帧”
> 消费模型，遇到 B 帧重排/深流水时会丢帧（与 FFmpeg 软解路径同一限制）。

## 注册与选择

每个能力由根目录工厂（`inference_engine_factory` / `detection_inference_engine_factory` /
`pose_estimation_factory` / `action_recognition_factory` / `video_decoder_factory` /
`video_encoder_factory` / `image_accelerator_factory`）持有后端注册表：

- 后端在编译期按 `WITH_*` 选项编入，运行期通过静态注册宏注册到工厂
- `AUTO` 模式按固定优先级选择第一个 `isAvailable()` 为真的后端
  （TensorRT > RKNN > Ascend > 地平线 BPU > OpenVINO > CPU）
- 硬件后端（RKNN/NVDEC/MPP/VPU…）通过 **dlopen 惰性加载**：x86 编译主机
  无对应库时 `isAvailable()` 返回 false，自动回退下一优先级；VAAPI 由 FFmpeg 内部
  加载 libva/驱动，本工程不直接链接 libva；OpenVINO 启用时链接 `libopenvino_c`（C ABI）
- 动态库加载统一使用 `include/ai_stream/hal/dl_library.h` 的 RAII `DlLibrary`（进程级共享、
  引用计数、线程安全，最后一个引用释放时 `dlclose`）；RKNN / MPP / RGA 后端已迁移，
  地平线 VPU/BPU 后端待板端 SDK 环境跟进
- 编码器（`video_encoder_factory`）的 `AUTO` 依次尝试 **`horizon_h264`（VPU）→
  `mpp_h264`（RK）→ `vaapi_h264`（Intel/AMD）→ `dvpp_h264`（Ascend）→
  `ffmpeg_h264`（软编/nvenc）**；
  sink/evidence 配置里写 `"hw_encoder": true` 即自动选硬件（false 为软编），无需手写后端名；
  也可用 `"encoder": "vaapi_h264"` / `"dvpp_h264"` / `"h264_nvenc"` 等显式指定
- 未显式设置 `output_width/output_height`（或 `output_size`）时，sink 延迟到首帧初始化，
  直接使用输入视频的宽高；显式设置时 `EncoderBase` 会把不同尺寸的输入缩放到目标尺寸
- 解码节点支持 `"hw_decoder_type": "vaapi"`（或 `h264_vaapi`/`qsv`/`nvdec`/`mpp`/
  `dvpp`/`horizon`/`ffmpeg`）显式指定后端；不写则 AUTO。AUTO 下硬件后端初始化失败
  会自动回退软件解码

## 目录结构

```
src/hal/
├── ascend/       # ACL 进程级初始化辅助（检测推理 / DVPP 解码编码共用）
├── infer/        {nvidia, intel, rk, ascend, horizon, cpu}   推理引擎
├── encode/       {rk, vaapi, ascend, horizon, cpu}            编码（MPP / VAAPI / DVPP / VPU / FFmpeg）
├── decode/       {nvidia, vaapi, rk, ascend, horizon, cpu}   解码
├── image_accel/  {nvidia, rk, ascend, horizon, cpu}          图像加速
└── *_factory.cpp                                             能力工厂

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
| 独立 NVENC HAL（`encode/nvidia/`） | 当前 NVENC 经 FFmpeg（`h264_nvenc`）实现；独立直驱 NvEncodeAPI 用于 GPU 帧零拷贝输入。**暂缓**：sink 编码输入为 OSD 输出的 host BGR，零拷贝输入需 OSD 输出留显存（卡在 GPU 文字渲染） | NVIDIA Video Codec SDK 头文件 + OSD GPU 化 |
| Ascend 姿态/通用推理 | CANN 推理引擎补齐（检测/动作识别已实现） | 昇腾设备 + CANN SDK |
| CPU 检测/姿态后端 | 补全 CPU 矩阵（当前节点 mock 兜底） | 低优先级 |