# docker — JetPack 7.2 镜像

本目录提供基于 **JetPack 7.2（L4T r39.2 / Orin）** 的构建与运行镜像，源码对应
`fix/jp72` 分支。

| 文件 | 说明 |
|---|---|
| `Dockerfile.jp72` | Ubuntu 24.04 + Jetson apt 源（CUDA 13.2 / TensorRT 10.16 / cuDNN）+ OpenCV 4.6 + FFmpeg + GStreamer，编译并安装到 `/opt/ai_stream` |
| `build-jp72.sh` | 构建镜像的封装脚本 |
| `run-jp72.sh` | 启动容器（GPU、编解码设备、源码挂载） |
| `build-in-container.sh` | 容器内 `build-ai-stream` 命令：对挂载的源码增量编译 |

## 前提

- aarch64 主机（Orin 等），已安装 Docker 与 NVIDIA Container Runtime
- 主机 JetPack 7.2 驱动正常（`nvidia-smi` 可用）
- 需要能访问 Ubuntu ports 与 NVIDIA Jetson apt 源（r39.2）
- **源码依赖已内置**：`3rd_party/spdlog`、`3rd_party/httplib/httplib.h`，
  配置阶段不再访问 GitHub

## 离线部署

网络受限环境建议在联网机器上构建一次，再分发镜像：

```bash
# 联网机器
docker/build-jp72.sh
docker save ai_stream_pipeline:jp72 | gzip > ai_stream_jp72.tar.gz

# 目标机器
gunzip -c ai_stream_jp72.tar.gz | docker load
```

目标机器只需运行容器（`docker/run-jp72.sh`），无需联网或重新编译。

## 构建

```bash
# 在仓库根目录
docker/build-jp72.sh

# 自定义镜像名 / 并行度 / CMake 参数
IMAGE=my/ai_stream:jp72 JOBS=8 docker/build-jp72.sh
CMAKE_ARGS="-DWITH_CUDA=ON -DWITH_TENSORRT=ON -DWITH_FTP=OFF -DWITH_VLM=OFF" \
    docker/build-jp72.sh
```

首次构建会下载 CUDA Toolkit 等依赖，耗时较长（Orin 上约 20–40 分钟），镜像约 10+ GB。

## 运行

```bash
# 进入容器 shell（源码挂载在宿主机相同绝对路径，可直接增量编译）
docker/run-jp72.sh

# HTTP 服务
docker/run-jp72.sh /opt/ai_stream/bin/http_server 0.0.0.0 8080

# 示例管道
docker/run-jp72.sh /opt/ai_stream/bin/simple_detection \
    /home/nvidia/lj/ai_stream_pipeline/build/api_state/templates/child_intrusion_fighting_fusion.json
```

`run-jp72.sh` 会：

- `--gpus all`，并挂载 JP7 的 `/opt/nvidia/l4t-gpu-libs/nvgpu`（若存在）
- 挂载 `/dev/v4l2-nvdec`、`/dev/v4l2-nvenc`、`/dev/nvmap`、`/dev/dri`（存在才挂）
- 把仓库根目录挂到容器内**相同路径**，这样管道 JSON 里的绝对模型路径无需修改
- 以当前用户 UID/GID 运行，避免在挂载目录生成 root 属主文件
- `--net host`，方便 RTSP/RTMP/HTTP 访问

可覆盖的环境变量：`IMAGE`、`REPO_ROOT`、`DOCKER_USER`，以及透传
`VLM_API_KEY`、`AI_STREAM_API_TOKEN`、`AI_STREAM_STATE_DIR`。

## 模型与配置

- 模型（`models/`）和引擎文件**不打进镜像**（`.dockerignore` 已排除），
  由 `run-jp72.sh` 挂载仓库目录使用；换机器时把 `models/` 一起带过去即可。
- 若模型不在仓库目录，修改管道 JSON 里的 `model_path`，或自行追加
  `-v /your/models:/your/models`（在 `run-jp72.sh` 中扩展）。
- 仓库内 `config/` 可直接使用；程序按**当前工作目录**读取 `config/`，
  `run-jp72.sh` 默认把工作目录设为挂载的仓库根目录。

## 容器内重新编译

```bash
docker/run-jp72.sh
# 容器内：
build-ai-stream                  # Release，输出到 /src/build，并安装到 /opt/ai_stream
build-ai-stream -DWITH_FTP=OFF   # 追加参数
```

## 中文面板字体

镜像内 OpenCV 固定为 Ubuntu 24.04 的 **4.6**（自带 `opencv_freetype`），
`freetype` 可用，`resources/ai.ttf` 能正常渲染中文；不会被 Jetson 源的
OpenCV 4.8（无 contrib）抢占。

## 关于 nvv4l2 硬解/硬编

镜像通过 Jetson `common` + `som` 源安装：

- `nvidia-l4t-gstreamer`：提供 `nvv4l2decoder` / `nvvidconv` / `nvv4l2h264enc`
- `nvidia-l4t-jetson-multimedia-api`：提供 `v4l2_nv_extensions.h`（裸 V4L2 解码后端）

安装前会创建 `/opt/nvidia/l4t-packages/.nv-l4t-disable-boot-fw-update-in-preinstall`，
跳过 `nvidia-l4t-core` 对 `/proc/device-tree/compatible` 的检查（`docker build` 环境没有该节点）。

运行时确保映射设备节点（`run-jp72.sh` 自动处理）：

- `/dev/v4l2-nvdec`、`/dev/v4l2-nvenc`、`/dev/nvmap`
- `/opt/nvidia/l4t-gpu-libs/nvgpu`（JP7 的真实 `libcuda.so.1`）

验证：

```bash
docker/run-jp72.sh bash -lc "gst-inspect-1.0 nvv4l2decoder | head -3; gst-inspect-1.0 nvv4l2h264enc | head -3"
```

如果设备/驱动未映射，程序会自动回退 FFmpeg 软解 / libx264 软编。

## 备注

- 本镜像仅适用于 aarch64 + JetPack 7.2；x86 请直接在宿主机编译。
- 日志、证据、`api_state/` 写在挂载目录，持久化在宿主机。
