# 编译规则

## 1. 环境要求

- CMake >= 3.16，C++17 编译器（GCC/Clang）
- 必需：FFmpeg（avcodec/avformat/avutil/swscale）、OpenCV（core/imgproc/videoio）、spdlog、nlohmann_json
- 可选：CUDA Toolkit、TensorRT、RKNN（RK3588）、Ascend CANN、Eigen3（跟踪）、libcurl（FTP）、GTest（测试）

依赖解析策略：

- **spdlog**：优先使用系统安装版本；找不到时自动 FetchContent 从 GitHub 拉取 v1.9.2
  （不使用 v1.12.0：其捆绑的 fmt v9 与 GCC 11/C++17 存在 constexpr 兼容性问题）
- **OpenCV freetype**：可选组件。检测到 `opencv_freetype` target 时定义
  `HAVE_OPENCV_FREETYPE` 宏并链接；缺失时代码自动回退 `cv::putText`（中文绘制受限），
  仅输出 STATUS 提示、不会导致配置失败
- **cpp-httplib**：BUILD_HTTP_SERVER=ON 时 FetchContent 自动拉取

## 2. CMake 选项一览

```bash
cmake -LH build   # 查看全部选项及说明
```

### 构建/功能开关

| 选项 | 默认 | 说明 |
|---|---|---|
| `BUILD_HTTP_SERVER` | ON | HTTP API 服务可执行文件 |
| `BUILD_TESTS` | OFF | 单元测试（需 GTest） |
| `BUILD_EXAMPLES` | ON | 示例程序 |
| `BUILD_TOOLS` | ON | bench 性能基准工具 |
| `ENABLE_COVERAGE` | OFF | 覆盖率插桩（-fprofile-arcs -ftest-coverage） |
| `INSTALL_HEADERS` | ON | 安装公开头文件 |

### 平台后端开关

| 选项 | 默认 | 说明 |
|---|---|---|
| `WITH_CUDA` | OFF | GPU 加速（解码/预处理/绘制），启用 CUDA 语言 |
| `WITH_TENSORRT` | OFF | TensorRT 推理后端 |
| `WITH_NPP` | OFF | NPP 图像处理库 |
| `WITH_RKNN` | OFF | RK3588 推理后端 |
| `WITH_ASCEND` | OFF | 昇腾 CANN 推理后端 |
| `WITH_CPU_FALLBACK` | ON | CPU fallback 实现 |
| `WITH_TRACK` | ON | 跟踪节点（需 Eigen3） |
| `WITH_ALERT` | ON | 告警节点 |

可选依赖查找失败时**自动禁用对应功能并输出 WARNING**（如 TensorRT 未找到则
`WITH_TENSORRT` 自动置 OFF），不会中断配置。

## 3. 典型平台配置

```bash
# x86 + NVIDIA GPU（按显卡指定架构：T4=75, A100=80, 30xx=86, 40xx=89, H100=90）
cmake -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON \
      -DCMAKE_CUDA_ARCHITECTURES="86" \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc

# x86 纯 CPU
cmake -B build

# RK3588
cmake -B build -DWITH_RKNN=ON

# 昇腾
cmake -B build -DWITH_ASCEND=ON

# 开发/CI 全量构建
cmake -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON -DBUILD_TESTS=ON \
      -DENABLE_COVERAGE=ON
```

## 4. 构建与测试

```bash
cmake --build build -j$(nproc)

# 全量测试（ctest）
ctest --test-dir build --output-on-failure

# 或单独运行
./build/tests/unit/test_core
./build/tests/unit/test_nodes
```

## 5. 构建产物

| 产物 | 路径 |
|---|---|
| HTTP 服务 | `build/src/http/http_server` |
| 示例服务 | `build/examples/http_server/ai_stream_server` |
| 核心库 | `build/src/core/libai_stream_core.so` |
| HAL 库 | `build/src/hal/libai_stream_hal.a` |
| 节点库 | `build/src/nodes/libai_stream_nodes.so` |
| 基准工具 | `build/tools/benchmark/bench` |
| 测试 | `build/tests/unit/test_core`、`test_nodes` |

## 6. 注意事项

- 不要全局添加 `-Wl,--allow-multiple-definition`：会掩盖真实的符号重复问题，
  链接报错时应定位根因
- CUDA 架构必须显式匹配目标硬件，错误的架构会导致运行时无法加载 kernel
- Release 为默认构建类型；修改 CMakeLists.txt 后建议清理
  `build/CMakeCache.txt` 再重新配置
