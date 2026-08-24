ai_stream_pipeline/
├── CMakeLists.txt                 # 顶层构建脚本（选项/依赖/库/安装规则）
├── cmake/                         # CMake 辅助模块
│   ├── dependencies.cmake         # 统一依赖查找
│   ├── FindFFmpeg.cmake           # 随 install 导出，供下游 find_dependency
│   ├── FindTensorRT.cmake
│   ├── FindRKNN.cmake             # RK3588 平台
│   ├── FindAscend.cmake           # 昇腾平台
│   ├── FindEigen3.cmake
│   ├── Findnlohmann_json.cmake
│   └── ai_stream_pipelineConfig.cmake.in  # 安装包配置模板
│
├── include/                       # 【公开头文件目录】对外 API 边界
│   └── ai_stream/
│       ├── core/                  # 核心框架
│       │   ├── node.h             # 基础节点抽象类（推模式 + configure）
│       │   ├── queued_node.h      # 带输入队列的异步节点混入（背压 + STREAM_END 自停）
│       │   ├── bounded_queue.h    # 线程安全有界阻塞队列
│       │   ├── packet.h           # 数据包基类/告警事件/枚举映射表
│       │   ├── pipeline.h         # 管道管理器（拓扑启停 + 环检测）
│       │   └── metrics.h          # 指标收集单例
│       ├── hal/                   # 硬件抽象层接口
│       │   ├── i_inference_engine.h / i_detection_inference_engine.h
│       │   ├── i_action_recognition.h
│       │   ├── i_image_accelerator.h      # 图像处理加速（预处理/绘制/NMS）
│       │   ├── i_video_codec.h            # 视频编解码抽象
│       │   └── *_factory.h                # 各后端工厂（按平台选择实现）
│       ├── nodes/                 # 节点插件接口
│       │   ├── i_source_node.h    # 拉流/文件源（含通用 configure）
│       │   ├── i_decode_node.h
│       │   ├── i_preprocess_node.h / i_gpu_preprocess_node.h
│       │   ├── i_infer_node.h
│       │   ├── i_postprocess_node.h / i_gpu_postprocess_node.h
│       │   ├── i_tracker_node.h
│       │   ├── i_action_recognition_node.h
│       │   ├── i_fusion_node.h
│       │   ├── i_draw_node.h      # OSD 绘制（可选依赖 OpenCV freetype，宏 HAVE_OPENCV_FREETYPE）
│       │   ├── i_evidence_node.h
│       │   └── i_sink_node.h
│       └── rules/
│           └── i_alert_rule.h     # 告警规则接口
│
├── src/                           # 【内部实现】
│   ├── core/
│   │   ├── node.cpp
│   │   ├── pipeline_manager.cpp   # Pipeline::buildFromJson/拓扑启停/自停后重启恢复
│   │   ├── async_pipeline_manager.h/.cpp  # 异步管道管理（任务队列/资源监控/auto_batch）
│   │   └── metrics.cpp
│   ├── hal/                       # 硬件抽象层各后端实现
│   │   ├── tensorrt/              # TensorRT 检测/通用推理/动作识别引擎
│   │   ├── rknn/                  # RK3588 推理后端
│   │   ├── ascend/                # 昇腾 CANN 后端
│   │   ├── cpu/                   # CPU fallback（image accelerator 等）
│   │   ├── ffmpeg/ nvdec/ mpp/ npp/ rga/  # 编解码与图像加速后端
│   │   └── *_factory.cpp          # 工厂：运行时按编译选项选择后端
│   ├── http/                      # REST API 服务
│   │   ├── api_server.h/.cpp      # 同步/异步双模式 handler
│   │   ├── handlers/              # 具体路由 handler
│   │   └── main.cpp               # 入口（--async 开关、logging.json 加载、信号处理）
│   ├── nodes/
│   │   ├── registry/              # 节点工厂（REGISTER_NODE 宏）
│   │   ├── source/                # rtsp_source / file_source
│   │   ├── decode/                # ffmpeg_decode / decoder_pool / hw_cuda_decode
│   │   ├── preprocess/            # resize_normalize（CPU/GPU/CUDA 三版）
│   │   ├── infer/                 # detection_infer / pose_infer / cuda_pose_infer /
│   │   │                          # action_recognition_videomae / rknn_detection_infer /
│   │   │                          # int8_calibrator
│   │   ├── postprocess/           # detection_post / gpu_detection_post
│   │   ├── track/                 # tracker_node + ocsort/bytetrack 适配器 + GPU Kalman
│   │   ├── alert/                 # alert_node（规则容器，并行/串行）
│   │   ├── fusion/                # fusion_node（动作+检测融合，跨源 NMS 去重）
│   │   ├── draw/                  # osd_draw / gpu_osd_draw / draw_panel（告警面板）
│   │   ├── evidence/              # evidence_node / frame_buffer / video_recorder /
│   │   │                          # video_rollover / ftp_uploader
│   │   └── sink/                  # rtmp_sink / mp4_save / encoder_base
│   └── rules/alert/               # 20+ 告警规则 + alert_rule_factory
│       └── detector/              # 攀爬/打架/打电话等复合检测器
│
├── 3rd_party/                     # 第三方依赖
│   ├── tracker/                   # OCSort / ByteTrack（独立库）
│   ├── log_mgr/                   # 日志管理（spdlog 封装）
│   └── Eigen/  nlohmann/          # 头文件库
│
├── config/                        # 【运行时配置】
│   ├── pipelines/                 # 管道拓扑 JSON（alert/fusion/evidence/benchmark 等）
│   ├── sources/camera_list.json   # 流地址预设
│   └── logging/logging.json       # 日志配置（main.cpp 加载）
│
├── tests/
│   ├── unit/core/                 # 单元测试（GTest）：队列/节点/管道生命周期/数据包
│   ├── unit/nodes/                # 节点测试：跟踪匹配/融合/文件源自停重启
│   ├── data/sample_5s.mp4         # 测试视频
│   └── CMakeLists.txt
│
├── examples/
│   ├── simple_detection/  multi_stream/  http_server/   # ai_stream_server 示例
│   └── custom_node/               # 自定义节点示例
│
├── tools/
│   ├── benchmark/bench.cpp        # 管道性能基准（跑定时长输出指标）
│   ├── model_converter/           # ONNX/TRT 转换与量化工具（Python）
│   ├── videomae_train/            # 动作识别训练脚本
│   └── train_climbing_svm.py
│
├── utils/                         # header-only 工具（随公开头文件安装）
│   ├── time_util.h  string_helper.h  zone_utils.h  cuda_check.h
│
├── models/                        # 模型文件（.engine/.onnx，不入库，建议 gitignore）
├── resources/                     # 字体/Logo/参考图（画框面板用）
└── docs/                          # 本文档目录（索引见 readme.md）
