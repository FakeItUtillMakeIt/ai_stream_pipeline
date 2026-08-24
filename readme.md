# ai_stream_pipeline 文档索引

模块化 AI 视频流处理框架：以**节点图（DAG）**组织处理逻辑，数据以 **Packet** 在节点间流动，
通过 HTTP API 动态构建、启停管道。支持 x86 GPU（CUDA/TensorRT）与 ARM 平台（RKNN/Ascend）多后端。

## 总体架构图

```
[HTTP Server] ---(POST JSON)---> [Pipeline Manager]
                                        |
                                        v
[RTSP/File Source] --(RawVideo)--> [Decode] --(YUV/BGR)--> [Preprocess]
                                                                 |
                    +--------------------------------------------+-------------+
                    |                                            |             |
                    v                                            v             v
           [Detection Infer]                            [Pose/Action Infer]  [Preview]
                    |                                            |
                    v                                            v
              [Tracker]                                   [Fusion Node]
                    |                                            |
                    +--------------------+-----------------------+
                                         |
                                         v
                                  [Alert Rules] --> [Draw OSD] --> [Evidence/RTMP Sink]
```

## 文档目录

| 文档 | 内容 |
|---|---|
| [architecture.md](./docs/architecture.md) | 架构设计说明：核心概念（Node/Packet/Pipeline）、库划分、数据流 |
| [structure.md](./docs/structure.md) | 项目目录结构：源码布局、HAL 硬件抽象层、各模块职责 |
| [compile.md](./docs/compile.md) | 编译规则：CMake 选项、平台配置组合、构建产物、测试 |
| [deployment.md](./docs/deployment.md) | 部署指南：依赖清单、运行 HTTP 服务、安装与集成、优雅退出 |
| [api_reference.md](./docs/api_reference.md) | HTTP API 参考（同步/异步模式、管道 JSON 格式） |
| [adding_new_node.md](./docs/adding_new_node.md) | 自定义节点开发指南 |
| [action_recognition_guide.md](./docs/action_recognition_guide.md) | 动作识别（VideoMAE）使用指南 |

## 快速开始

```bash
# x86 + NVIDIA GPU 环境
cmake -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON -DBUILD_TESTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES="86"
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure

# 启动服务
./build/src/http/http_server 0.0.0.0 8080
```

详细说明见 [compile.md](compile.md) 与 [deployment.md](deployment.md)。

## 支持的平台后端

| 后端 | CMake 选项 | 说明 |
|---|---|---|
| TensorRT | `WITH_TENSORRT` | NVIDIA 推理引擎（.engine 模型） |
| CUDA / NPP | `WITH_CUDA` / `WITH_NPP` | GPU 解码/预处理/绘制加速 |
| RKNN | `WITH_RKNN` | 瑞芯微 RK3588 推理 |
| Ascend CANN | `WITH_ASCEND` | 华为昇腾推理 |
| CPU fallback | `WITH_CPU_FALLBACK` | 无 GPU 时 CPU 实现（默认开启） |

所有后端通过 HAL（硬件抽象层，`src/hal/`）统一接口，缺失的可选依赖自动降级禁用，
文本渲染在无 OpenCV freetype 模块时自动回退到 `cv::putText`。
