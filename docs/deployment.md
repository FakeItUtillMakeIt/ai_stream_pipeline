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
| RKNN / Ascend CANN SDK | 嵌入式平台推理 | 对应平台时 |
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

cmake --build build -j$(nproc)
```

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

接口详见 [api_reference.md](api_reference.md)。

### 运行时配置

- **日志**：`config/logging/logging.json`（相对工作目录），可配级别/目录/滚动/异步队列；
  不存在时使用内置默认值
- **管道拓扑**：`config/pipelines/*.json`，通过 HTTP build 接口提交（请求体内联 JSON）
- **流地址预设**：`config/sources/camera_list.json`（运维参考数据）
- **模型**：TensorRT `.engine` 文件路径写在各管道 JSON 的 `detector_config.model_path`，
  模型转换工具见 `tools/model_converter/`

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
