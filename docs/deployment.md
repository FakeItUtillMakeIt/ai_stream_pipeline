# 部署指南

## 1. 依赖

| 依赖 | 用途 | 必需 |
|---|---|---|
| FFmpeg (avcodec/avformat/avutil/swscale) | 拉流/解码/编码 | 是 |
| OpenCV (core/imgproc/videoio) | 图像处理 | 是 |
| OpenCV freetype 模块 | 中文绘制（缺失时自动回退 `cv::putText`，中文显示受限） | 可选 |
| spdlog / nlohmann_json | 日志 / JSON（系统无 spdlog 时构建自动拉取 v1.9.2） | 是 |
| CUDA Toolkit + cuDNN | GPU 加速（解码/预处理/推理/绘制） | WITH_CUDA=ON 时 |
| TensorRT | 推理引擎（.engine 模型） | WITH_TENSORRT=ON 时 |
| VAAPI 驱动（iHD/Mesa + 内核 i915/xe） | Intel/AMD 硬解/硬编 | WITH_VAAPI=ON 时（默认） |
| OpenVINO Runtime（libopenvino_c） | Intel CPU/iGPU/NPU 检测推理 | WITH_OPENVINO=ON 时 |
| Intel compute runtime（intel-opencl-icd / Level Zero） | OpenVINO 使用 iGPU/NPU | 仅 iGPU/NPU 推理时 |
| RKNN / Ascend CANN SDK | 嵌入式平台推理 | 对应平台时 |
| Ascend 驱动 + HwHiAiUser 组权限 | DVPP 硬解 / ACL 检测推理 | WITH_ASCEND=ON 时 |
| CANN `set_env.sh`（driver 运行库路径） | Ascend 运行时 LD_LIBRARY_PATH | WITH_ASCEND=ON 时 |
| 地平线 DNN / spcdev（RDK S100P） | BPU 推理 / VPU 编解码 | WITH_HORIZON=ON 时 |
| Eigen3 | 跟踪器（OCSort/ByteTrack） | WITH_TRACK=ON 时 |
| libcurl | 证据 FTP 上传 | 可选（缺失则禁用 FTP） |
| GTest | 单元测试 | BUILD_TESTS=ON 时 |
| cpp-httplib | HTTP 服务（FetchContent 自动拉取） | BUILD_HTTP_SERVER=ON 时 |

## 2. 构建

```bash
# 3090/A100 等 NVIDIA GPU（按目标卡指定 CUDA 架构：75/80/86/89/90）
cmake -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON \
      -DCMAKE_CUDA_ARCHITECTURES="86"

# WSL / 纯 CPU 环境
cmake -B build

# Intel/AMD iGPU：VAAPI 硬解/硬编 + OpenVINO 推理
cmake -B build -DWITH_VAAPI=ON -DWITH_OPENVINO=ON \
      -DOPENVINO_ROOT=/opt/intel/openvino/runtime

# 华为昇腾：ACL 检测推理 + DVPP 硬解
cmake -B build -DWITH_ASCEND=ON

cmake --build build -j$(nproc)
```

> 昇腾运行前置：
> ```bash
> sudo usermod -aG HwHiAiUser $USER          # 访问 /dev/davinci*
> id -nG | grep HwHiAiUser
> npu-smi info
> # 检测模型先转 .om（需要 ATC）：
> tools/model_converter/ascend/convert_onnx_to_om.sh \
>     models/sevncevision/20260921_sevnce_9_cls_nms_trans_640.onnx \
>     images 640 640 Ascend310P3
> # 动作识别：
> INPUT_SHAPE="input:1,16,3,224,224" \
> tools/model_converter/ascend/convert_onnx_to_om.sh \
>     models/action_recognition/action_recognition.onnx input 224 224 Ascend310P3
> ```
> 构建产物已通过 RPATH 指向 CANN lib64/driver 目录，通常无需手动
> `source set_env.sh`；若移动了 CANN 安装位置，则运行前 source
> `/usr/local/Ascend/ascend-toolkit/set_env.sh`。
> 检测节点用 `"backend": "ascend"` + `.om`，解码节点用
> `"hw_decoder": true, "hw_decoder_type": "dvpp"`（或 AUTO）；
> sink/evidence 写 `"hw_encoder": true`（或 `"encoder": "dvpp_h264"`）走 DVPP VENC。
> 未显式设置 `output_width/output_height` 时，sink 使用首帧输入视频宽高。

> Intel 核显运行前置：
> ```bash
> sudo apt install intel-media-va-driver vainfo            # VAAPI 驱动
> sudo usermod -aG render $USER                            # 允许访问 /dev/dri/renderD128
> sudo apt install intel-opencl-icd                        # OpenVINO GPU 插件依赖（可选）
> vainfo --display drm --device /dev/dri/renderD128          # 验证 H264/HEVC 编解码
> ```
> 没有 `render` 权限或驱动时，编解码会自动回退 FFmpeg 软解/软编；
> OpenVINO 会从 GPU 回退 CPU，均不影响服务启动。

完整选项与平台配置组合见 [compile.md](compile.md)。

产物：`build/src/http/http_server`、`build/src/core/libai_stream_core.so`、
`build/src/nodes/libai_stream_nodes.so`、`build/tools/benchmark/bench`。

## 3. 运行 HTTP 服务

```bash
cd <工作目录>   # 需包含 config/，且对 logs/ 有写权限

# 同步模式（默认）：build/start/stop 请求阻塞至完成
./build/src/http/http_server 0.0.0.0 8080

# 异步模式：请求提交任务立即返回 202，状态轮询
./build/src/http/http_server 0.0.0.0 8080 --async
```

> 生产环境建议：绑定内网/回环地址，并通过环境变量 `AI_STREAM_API_TOKEN=<token>` 开启
> Bearer 鉴权（未设置则不鉴权）。请求限制/错误码详见 [api_reference.md](api_reference.md)。

接口详见 [api_reference.md](api_reference.md)。

### 运行时配置

- **日志**：`config/logging/logging.json`（相对工作目录），可配级别/目录/滚动/异步队列；
  不存在时使用内置默认值
- **管道拓扑**：`config/pipelines/*.json`，通过 HTTP build 接口提交（请求体内联 JSON）
- **流地址预设**：`config/sources/camera_list.json`（运维参考数据）
- **模型**：TensorRT `.engine` / 地平线 `.hbm` / RKNN `.rknn` 文件路径写在各管道 JSON 的
  `detector_config.model_path`；模型转换工具见 `tools/model_converter/`
  （地平线见 `tools/model_converter/horizon/`）
- **硬件编码**：sink/evidence 配置写 `"hw_encoder": true` 自动选平台硬件编码器
  （地平线 VPU / RK MPP / NVENC），无硬件时自动回退软编；也可用 `"encoder": "..."` 显式指定

> 注意：
> - 敏感凭据（如 FTP 密码）不要写入仓库内的管道配置
> - 配置中的 `reference_image`、字体等路径建议使用相对路径或部署时统一调整，
>   避免硬编码开发机绝对路径

## 4. 性能基准

```bash
# 无 RTSP 流时可用 file_source 管道压测（tests/data/sample_5s.mp4）
./build/tools/benchmark/bench config/pipelines/file_decode_benchmark.json 30
# 输出各节点 total_packets / dropped_packets / latency / fps
```

## 5. 安装与集成

```bash
cmake --install build --prefix /opt/ai_stream
```

安装内容：库、http_server/bench、公开头文件（含 `utils/`）、
CMake 包配置（`lib/cmake/ai_stream_pipeline/`）、`config/` 模板。

下游工程集成：

```cmake
find_package(ai_stream_pipeline REQUIRED)
target_link_libraries(app PRIVATE ai_stream_pipeline::core)
# 需要节点工厂注册时再链接 ai_stream_pipeline::nodes
```

## 6. 测试

```bash
cmake -B build -DBUILD_TESTS=ON
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure    # 或 ./build/tests/unit/test_core
```

## 7. 优雅退出

服务捕获 SIGINT/SIGTERM：停止 HTTP 监听 → 停止全部管道（拓扑序）→ 刷新日志。
异步模式下 AsyncPipelineManager 会先排空任务队列再销毁。
管道因 STREAM_END 自停后可直接重新 start，无需先 stop 再 start。

## 8. 地平线 RDK S100P 部署

1. **板端本机构建**（见 [compile.md](compile.md)）：
   `cmake -B build_horizon -DWITH_HORIZON=ON`

2. **转换模型**（x86 Docker，HBDK 工具链）：
   ```bash
   cd tools/model_converter/horizon
   ./convert.sh ./model.onnx --imgsz 640 --march nash-m \
       --calibration ./calibration_dataset/ --int16
   ```
   要点：ONNX 用**静态形状**导出（`export_onnx.py`）；校准集用与板端一致预处理的 `.npy`
   （`make_calib_npy.py`）；int8 可能使分类头量化塌缩，建议 `--int16`。

3. **运行**：
   ```bash
   LD_LIBRARY_PATH=/usr/hobot/lib:$LD_LIBRARY_PATH \
     ./build_horizon/examples/simple_detection/simple_detection config/pipelines/fusion_pipeline_v2.json
   ```
   - `fusion_pipeline_v2.json`：RGB int8 路径
   - NV12 直通路径：模型为 NV12 双输入时，配置 `decode.output_nv12` + `infer.input_nv12` 并
     让 `decode → infer` 直连（`decode` 设 `output_nv12`、`infer` 设 `input_nv12`）
   - 编码由 `"hw_encoder": true` 自动使用 VPU 硬编
