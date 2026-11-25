# docker — JetPack 7.2 镜像

本目录提供基于 **JetPack 7.2（L4T r39.2 / Orin）** 的构建与运行镜像，源码对应
`fix/jp72` 分支。

| 文件 | 说明 |
|---|---|
| `Dockerfile.jp72` | Ubuntu 24.04 + Jetson apt 源（CUDA 13.2 / TensorRT 10.16 / cuDNN）+ OpenCV 4.6 + FFmpeg + GStreamer，编译并安装到 `/opt/ai_stream` |
| `build-jp72.sh` | 构建镜像的封装脚本（自动执行 stage-assets） |
| `stage-assets.sh` | 把模型/mediamtx 暂存到 `docker/assets/`（不入库） |
| `run-jp72.sh` | 启动容器（GPU、编解码设备、源码挂载） |
| `assets/` | 测试视频、demo 管道配置、mediamtx 配置、启动脚本 |
| `build-in-container.sh` | 容器内 `build-ai-stream` 命令：对挂载的源码增量编译 |

## 前提

- aarch64 主机（Orin 等），已安装 Docker 与 NVIDIA Container Runtime
- 主机 JetPack 7.2 驱动正常（`nvidia-smi` 可用）
- 需要能访问 Ubuntu ports、NVIDIA Jetson apt 源（r39.2）以及 npm registry
  （用于构建前端；可用 `--build-arg NPM_REGISTRY=https://registry.npmmirror.com` 切换）
- 模型文件需先 `git lfs pull`（`stage-assets.sh` 会校验是否为真实文件而不是 LFS 指针）
- mediamtx 二进制默认从 `/home/nvidia/lj/pkg/mediamtx` 复制，缺失时按
  `MEDIAMTX_VER=1.1.0` 从 GitHub release 下载
- **源码依赖已内置**：`3rd_party/spdlog`、`3rd_party/httplib/httplib.h`，
  CMake 配置阶段不再访问 GitHub

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

> 直接用 `docker build -f docker/Dockerfile.jp72 .` 时，需要先执行
> `docker/stage-assets.sh` 准备模型与 mediamtx；`docker/build-jp72.sh` 会自动执行。

## 运行（不挂载仓库，镜像已自包含）

镜像内置：

- 4 个引擎：`yoloe_v8s_child_wall_gate`、`20260921_sevnce_9_cls_nms_trans_640`、
  `relateanything`、`action_recognition.rebuild`
- relateanything 的 `predicate_bank.json` 等配置、`resources/ai.ttf`、`logo.png`
- `scripts/`、demo 管道配置、测试视频 `child_on_wall.mp4`
- `mediamtx`（RTMP 1935 / HLS 8888）
- 前端 dist（`/opt/ai_stream/web`）

默认 CMD 会同时启动 mediamtx 和 http_server（前端与 API 同源 8080）：

```bash
# 不需要挂载任何东西（带接口鉴权 token）
AI_STREAM_API_TOKEN=yanfa AI_STREAM_IMPL_TOKEN=shishi \
    NO_MOUNT=1 HW_CODEC=1 docker/run-jp72.sh

# 或直接 docker run（不使用 run 脚本）
docker run --rm --runtime nvidia --privileged -v /dev:/dev \
    -v /opt/nvidia/l4t-gpu-libs/nvgpu:/opt/nvidia/l4t-gpu-libs/nvgpu:ro \
    -v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro \
    --net host --ipc host ai_stream_pipeline:jp72
```

浏览器访问 `http://<IP>:8080/`。

直接跑内置 demo 管道（同时拉起 mediamtx）：

```bash
NO_MOUNT=1 HW_CODEC=1 docker/run-jp72.sh /usr/local/bin/run-demo
```

## 运行（挂载仓库 / 开发）

```bash
# 进入容器 shell（仓库挂载在宿主相同绝对路径）
docker/run-jp72.sh bash

# 挂载仓库并启动服务
docker/run-jp72.sh /usr/local/bin/start-ai-stream
```

`run-jp72.sh` 会：

- `--gpus all`，并挂载 JP7 的 `/opt/nvidia/l4t-gpu-libs/nvgpu`（若存在）
- 挂载 `/dev/v4l2-nvdec`、`/dev/v4l2-nvenc`、`/dev/nvmap`、`/dev/dri`（存在才挂）
- 把仓库根目录挂到容器内**相同路径**，这样管道 JSON 里的绝对模型路径无需修改
- 以当前用户 UID/GID 运行，避免在挂载目录生成 root 属主文件
- `--net host`，方便 RTSP/RTMP/HTTP 访问

可覆盖的环境变量：`IMAGE`、`REPO_ROOT`、`DOCKER_USER`，以及透传
`VLM_API_KEY`、`AI_STREAM_API_TOKEN`、`AI_STREAM_STATE_DIR`。

如果管道引用的视频/模型不在仓库目录内（例如 `file:///home/nvidia/lj/child_on_wall.mp4`），
用 `MOUNT_DIR` 额外挂载该目录（容器内保持相同绝对路径）：

```bash
MOUNT_DIR=/home/nvidia/lj HW_CODEC=1 docker/run-jp72.sh \
    /opt/ai_stream/bin/simple_detection \
    /home/nvidia/lj/ai_stream_pipeline/build/api_state/templates/child_intrusion_fighting_fusion.json
```

## 模型与配置

- 镜像内置 4 个 demo 引擎（yoloe / sevncevision 检测 / relateanything /
  action_recognition）及 relateanything 的 `predicate_bank.json` 等配置，
  路径统一为 `/opt/ai_stream/models/...`。
- 测试视频：`/opt/ai_stream/assets/child_on_wall.mp4`。
- demo 管道：`/opt/ai_stream/config/pipelines/child_intrusion_fighting_fusion.demo.json`
  （已把所有绝对路径改成 `/opt/ai_stream/...`，并关闭 VLM/上报以便离线运行）。
- 挂载仓库时，仓库里的 `models/` 会覆盖/补充镜像内资源；修改管道 JSON 即可使用
  仓库里的其它模型。
- 程序按**当前工作目录**读取 `config/`；`NO_MOUNT=1` 时工作目录是 `/opt/ai_stream`，
  挂载仓库时是仓库根目录。

## 容器内重新编译

```bash
# 挂载仓库后
docker/run-jp72.sh bash
# 容器内（编译挂载进来的仓库，产物安装到 /opt/ai_stream）：
SRC="$PWD" build-ai-stream
SRC="$PWD" build-ai-stream -DWITH_FTP=OFF   # 追加参数
```

不挂载仓库时也可以直接编译镜像自带的 `/src`：`build-ai-stream`。

## 中文面板字体

镜像内 OpenCV 固定为 Ubuntu 24.04 的 **4.6**（自带 `opencv_freetype`），
`freetype` 可用，`resources/ai.ttf` 能正常渲染中文；不会被 Jetson 源的
OpenCV 4.8（无 contrib）抢占。

## Web 流水线编辑器

### 运行（不挂载仓库，推荐）

镜像在构建时已执行 `npm ci && npm run build`，产物在 `/opt/ai_stream/web`，
并设置 `AI_STREAM_WEB_ROOT=/opt/ai_stream/web`。`http_server` 会**同源托管**
前端页面，因此只需要启动一个进程：

```bash
# 默认 CMD 就是 start-ai-stream（mediamtx + http_server + 前端）
NO_MOUNT=1 HW_CODEC=1 docker/run-jp72.sh
```

浏览器直接访问 `http://<设备IP>:8080/`，前端与 `/api` 同源，无需 npm。

### 开发（挂载仓库）

镜像内保留 `/src` 源码、完整编译工具链、Node 18/npm 9 和前端 `node_modules`。
挂载仓库后可以编译后端：

```bash
docker/run-jp72.sh bash -lc 'cd "$PWD" && SRC="$PWD" build-ai-stream'
```

前端热更新（可选）：

```bash
docker/run-jp72.sh bash -lc \
  'cd "$PWD/web/pipeline-editor" && npm run dev -- --host 0.0.0.0'
```

浏览器访问 5173；Vite 会把 `/api` 代理到 `127.0.0.1:8080`。

- 挂载仓库时前端依赖直接复用仓库里的 `node_modules`；没有则 `npm install`。
- 使用镜像内自带的源码/依赖（不挂载仓库）时，路径是 `/src/web/pipeline-editor`。
- 构建镜像时可用 `NPM_REGISTRY=https://registry.npmmirror.com` 切换 npm 源：
  `docker build --build-arg NPM_REGISTRY=...`。

## 关于 nvv4l2 硬解/硬编

镜像通过 Jetson `common` + `som` 源安装：

- `nvidia-l4t-gstreamer`：提供 `nvv4l2decoder` / `nvvidconv` / `nvv4l2h264enc`
- `nvidia-l4t-jetson-multimedia-api`：提供 `v4l2_nv_extensions.h`（裸 V4L2 解码后端）

安装前会创建 `/opt/nvidia/l4t-packages/.nv-l4t-disable-boot-fw-update-in-preinstall`，
跳过 `nvidia-l4t-core` 对 `/proc/device-tree/compatible` 的检查。

注意：L4T 用户态库必须与宿主驱动版本严格匹配。默认模式下若无法初始化，
使用 **HW_CODEC 模式**（挂载宿主机的 `/usr/lib/aarch64-linux-gnu` 与 `/dev`）：

```bash
HW_CODEC=1 docker/run-jp72.sh /opt/ai_stream/bin/simple_detection \
    /home/nvidia/lj/ai_stream_pipeline/build/api_state/templates/child_intrusion_fighting_fusion.json
```

该模式会：

- `--privileged -v /dev:/dev`
- `-v /usr/lib/aarch64-linux-gnu:/usr/lib/aarch64-linux-gnu:ro`
- `LD_LIBRARY_PATH` 优先使用宿主的 `nvidia` 用户态库

代价是与宿主机用户态强耦合（镜像内同名库被覆盖），适合“同型号 Jetson 整机部署”。
默认模式（不设 `HW_CODEC`）只挂载必要设备，编解码不可用时自动回退软解/软编。

> TensorRT 已按宿主 JetPack 固定为 **10.16.2.10**（`TRT_PKG_VERSION` 可覆盖），
> 避免 som 源的 11.x 导致宿主机导出的 engine 无法在容器内反序列化。

## 备注

- 时区默认 `TZ=Asia/Shanghai`（`/etc/localtime` 指向上海），日志与证据文件名均为北京时间；
  如需覆盖：`TZ=UTC docker/run-jp72.sh ...`（run 脚本会透传 `TZ`）。
- 本镜像仅适用于 aarch64 + JetPack 7.2；x86 请直接在宿主机编译。
- 日志、证据、`api_state/` 写在挂载目录，持久化在宿主机。
