ai_stream_pipeline/
├── CMakeLists.txt                 # 顶层构建脚本
├── README.md                      # 项目概述与快速开始指南
├── .gitignore                     # 忽略 build/、.idea/、*.user 等
│
├── cmake/                         # CMake 辅助模块
│   ├── dependencies.cmake         # 统一依赖查找 (FetchContent/FindPackage)
│   ├── CompilerOptions.cmake      # 统一编译选项 (-Wall, -O3, C++17)
│   ├── FindFFmpeg.cmake           # 自定义 FFmpeg 查找脚本
│   └── FindTensorRT.cmake         # 自定义 TensorRT 查找脚本
│
├── include/                       # 【公开头文件目录】对外 API 边界
│   └── ai_stream/                 # 命名空间隔离，避免文件名冲突
│       ├── core/                  # 核心框架接口
│       │   ├── node.h             # 基础节点抽象类
│       │   ├── packet.h           # 数据包基类及常用派生类
│       │   ├── pipeline.h         # 管道管理器接口
│       │   └── thread_pool.h      # 线程池接口
│       └── nodes/                 # 节点插件接口（纯虚基类）
│           ├── i_source_node.h    # 拉流节点接口
│           ├── i_decode_node.h    # 解码节点接口
│           ├── i_infer_node.h     # 推理节点接口
│           ├── i_draw_node.h      # 画框节点接口
│           └── i_sink_node.h      # 推流/保存节点接口
│
├── src/                           # 【内部实现】不对外暴露
│   ├── core/                      # 核心框架实现
│   │   ├── node.cpp
│   │   ├── pipeline_manager.cpp
│   │   ├── thread_pool.cpp
│   │   └── frame_queue.h          # 内部使用的无锁队列 (仅供内部使用)
│   │
│   ├── nodes/                     # 具体功能节点实现
│   │   ├── registry/              # 【工厂注册机制】
│   │   │   ├── node_factory.h     # 工厂类与自动注册宏定义
│   │   │   └── node_registry.cpp  # 注册表单例实现
│   │   │
│   │   ├── source/                # 源节点
│   │   │   ├── rtsp_source.h
│   │   │   ├── rtsp_source.cpp
│   │   │   ├── file_source.h
│   │   │   └── file_source.cpp
│   │   │
│   │   ├── decode/                # 解码节点
│   │   │   ├── ffmpeg_decode.h
│   │   │   ├── ffmpeg_decode.cpp
│   │   │   └── decoder_pool.h     # 多路解码上下文池
│   │   │
│   │   ├── preprocess/            # 预处理节点 (可独立，也可融合)
│   │   │   ├── resize_normalize.h
│   │   │   └── resize_normalize.cpp
│   │   │
│   │   ├── infer/                 # 推理节点
│   │   │   ├── tensorrt_infer.h
│   │   │   ├── tensorrt_infer.cpp
│   │   │   ├── onnx_infer.h
│   │   │   └── onnx_infer.cpp
│   │   │
│   │   ├── postprocess/           # 后处理节点 (NMS、解析输出)
│   │   │   ├── detection_post.h
│   │   │   └── detection_post.cpp
│   │   │
│   │   ├── draw/                  # 画框节点
│   │   │   ├── osd_draw.h
│   │   │   └── osd_draw.cpp
│   │   │
│   │   └── sink/                  # 输出节点
│   │       ├── rtmp_sink.h
│   │       ├── rtmp_sink.cpp
│   │       ├── mp4_save.h
│   │       └── mp4_save.cpp
│   │
│   ├── http/                      # REST API 服务
│   │   ├── api_server.h
│   │   ├── api_server.cpp
│   │   └── handlers/              # 请求处理器
│   │       ├── pipeline_handler.cpp
│   │       └── status_handler.cpp
│   │
│   └── utils/                     # 内部通用工具
│       ├── logger.h               # 日志单例封装 (spdlog)
│       ├── string_helper.h
│       └── time_util.h
│
├── config/                        # 【运行时配置】
│   ├── pipelines/                 # 管道拓扑定义
│   │   ├── detection_pipeline.json
│   │   ├── multi_stream_pipeline.json
│   │   └── save_file_pipeline.json
│   ├── models/                    # 模型推理参数配置
│   │   ├── yolov8_config.json
│   │   └── resnet_config.json
│   ├── sources/                   # 预设流地址列表
│   │   └── camera_list.json
│   └── logging/                   # 日志配置
│       └── spdlog_config.toml
│
├── tests/                         # 【测试目录】
│   ├── unit/                      # 单元测试
│   │   ├── core/
│   │   │   ├── test_packet.cpp
│   │   │   └── test_pipeline.cpp
│   │   ├── nodes/
│   │   │   ├── test_decode.cpp
│   │   │   └── test_infer.cpp
│   │   └── CMakeLists.txt
│   ├── integration/               # 集成测试 (端到端)
│   │   ├── test_full_pipeline.cpp
│   │   └── CMakeLists.txt
│   └── data/                      # 测试数据 (小视频片段、模拟图片)
│       └── sample_5s.mp4
│
├── examples/                      # 【使用示例】
│   ├── simple_detection/          # 最简单检测示例
│   │   ├── main.cpp
│   │   └── CMakeLists.txt
│   ├── multi_stream/              # 多流并行示例
│   │   ├── main.cpp
│   │   └── CMakeLists.txt
│   └── custom_node/               # 演示如何开发自定义节点插件
│       ├── my_custom_node.h
│       ├── my_custom_node.cpp
│       └── CMakeLists.txt
│
├── tools/                         # 【开发辅助工具】
│   ├── benchmark/                 # 管道性能基准测试
│   │   ├── bench.cpp
│   │   └── CMakeLists.txt
│   └── model_converter/           # 模型转换脚本 (ONNX -> TRT)
│       ├── convert.py
│       └── README.md
│
└── docs/                          # 【项目文档】
    ├── architecture.md            # 架构设计说明
    ├── api_reference.md           # API 参考
    ├── adding_new_node.md         # 如何新增自定义节点
    └── deployment.md              # 部署指南