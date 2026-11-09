# src/hal/CMakeLists.txt
# 硬件抽象层（HAL）——按能力分目录：infer（推理）/ encode（编码）/
# decode（解码）/ image_accel（图像加速）/ common（共享工具），
# 能力目录内按厂商分子目录或前缀；根目录为各能力的工厂实现。

set(HAL_SOURCES
    inference_engine_factory.cpp
    detection_inference_engine_factory.cpp
    image_accelerator_factory.cpp
    video_decoder_factory.cpp
    action_recognition_factory.cpp
    relation_factory.cpp
    backend_diagnostics.cpp
)

# CPU fallback 始终编译
list(APPEND HAL_SOURCES
    image_accel/cpu/cpu_image_accelerator.cpp
    infer/cpu/cpu_inference_engine.cpp
    decode/cpu/ffmpeg_video_decoder.cpp
    pose_estimation_factory.cpp
    # 关系解码：与后端无关，CPU 兜底也要用，所以无条件编译
    infer/common/relation_decode.cpp
)

# Intel/AMD VAAPI 硬件编解码（纯 FFmpeg hwaccel 实现，不链接 libva；
# 运行时 device 不可访问时 isAvailable()=false，自动回退其它后端）
if(WITH_VAAPI)
    list(APPEND HAL_SOURCES
        decode/vaapi/vaapi_video_decoder.cpp
        encode/vaapi/vaapi_video_encoder.cpp
    )
endif()

# OpenVINO 检测推理（Intel CPU/iGPU/NPU，C API 避免 C++ ABI 绑定）
if(WITH_OPENVINO)
    list(APPEND HAL_SOURCES
        infer/intel/openvino_detection_engine.cpp
    )
endif()

# TensorRT 后端
if(WITH_TENSORRT)
    find_package(TensorRT REQUIRED)
    list(APPEND HAL_SOURCES
        infer/nvidia/trt_core.cpp
        infer/nvidia/tensorrt_inference_engine.cpp
        infer/nvidia/tensorrt_detection_engine.cpp
        infer/nvidia/tensorrt_action_recognition.cpp
        infer/nvidia/tensorrt_relation.cpp
        infer/nvidia/tensorrt_pose_estimation.cpp
    )
    # 姿态估计 GPU 预处理 kernel（需 CUDA 语言）
    if(WITH_CUDA)
        list(APPEND HAL_SOURCES infer/nvidia/tensorrt_pose_kernels.cu)
    endif()
endif()

if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND HAL_SOURCES
        image_accel/nvidia/npp_image_accelerator.cpp
        image_accel/nvidia/cuda_kernels_wrapper.cu
        decode/nvidia/nvdec_video_decoder.cpp
    )
    # Jetson V4L2 NVDEC/NVENC 后端（aarch64 平台，设备节点 + 头文件仅存在于 Jetson）
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        list(APPEND HAL_SOURCES
            decode/nvidia/v4l2_video_decoder.cpp
            decode/nvidia/gst_v4l2_video_decoder.cpp
            encode/nvidia/gst_v4l2_video_encoder.cpp
        )
    endif()
endif()

# RKNN 后端
if(WITH_RKNN)
    find_package(RKNN REQUIRED)
    list(APPEND HAL_SOURCES
        infer/rk/rknn_inference_engine.cpp
        infer/rk/rknn_detection_engine.cpp
        infer/rk/rknn_pose_estimation.cpp
        image_accel/rk/rga_image_accelerator.cpp
        decode/rk/mpp_video_decoder.cpp
        infer/rk/rknn_action_recognition.cpp
    )
endif()

# Ascend 后端
if(WITH_ASCEND)
    find_package(Ascend REQUIRED)
    list(APPEND HAL_SOURCES
        image_accel/ascend/dvpp_image_accelerator.cpp
        decode/ascend/dvpp_video_decoder.cpp
        encode/ascend/dvpp_video_encoder.cpp
        infer/ascend/ascend_action_recognition.cpp
        infer/ascend/ascend_detection_engine.cpp
    )
endif()

# Horizon BPU 后端（RDK S100P）
if(WITH_HORIZON)
    find_package(HorizonDNN REQUIRED)
    list(APPEND HAL_SOURCES
        infer/horizon/horizon_inference_engine.cpp
        infer/horizon/horizon_detection_engine.cpp
        infer/horizon/horizon_pose_estimation.cpp
        infer/horizon/horizon_action_recognition.cpp
        image_accel/horizon/horizon_image_accelerator.cpp
        decode/horizon/horizon_video_decoder.cpp
        encode/horizon/horizon_video_encoder.cpp
    )
endif()

# 编码后端与工厂（dlopen 运行时加载，所有平台编译）
list(APPEND HAL_SOURCES video_encoder_factory.cpp encode/rk/mpp_video_encoder.cpp encode/cpu/ffmpeg_video_encoder.cpp)

add_library(ai_stream_hal SHARED ${HAL_SOURCES})
set_target_properties(ai_stream_hal PROPERTIES EXPORT_NAME hal)

set_target_properties(ai_stream_hal PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(ai_stream_hal
    PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${PROJECT_SOURCE_DIR}
)

target_link_libraries(ai_stream_hal
    PUBLIC
        ai_stream_core
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_dnn
        FFmpeg::FFmpeg
)

# 条件链接推理后端
if(WITH_TENSORRT)
    target_compile_definitions(ai_stream_hal PRIVATE WITH_TENSORRT)
    target_include_directories(ai_stream_hal PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(ai_stream_hal PRIVATE ${TensorRT_LIBRARIES})
endif()

if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(ai_stream_hal PRIVATE WITH_CUDA)
    target_link_libraries(ai_stream_hal PRIVATE CUDA::cudart)
    # Jetson V4L2 后端需要 NVIDIA Multimedia API 扩展头文件 + GStreamer
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_include_directories(ai_stream_hal PRIVATE /usr/src/jetson_multimedia_api/include)
        find_package(PkgConfig REQUIRED)
        pkg_check_modules(GST REQUIRED gstreamer-1.0 gstreamer-app-1.0 gstreamer-base-1.0)
        target_include_directories(ai_stream_hal PRIVATE ${GST_INCLUDE_DIRS})
        target_link_libraries(ai_stream_hal PRIVATE ${GST_LIBRARIES})
        target_compile_options(ai_stream_hal PRIVATE ${GST_CFLAGS_OTHER})
    endif()
endif()

if(WITH_RKNN)
    target_compile_definitions(ai_stream_hal PRIVATE WITH_RKNN)
    target_include_directories(ai_stream_hal PRIVATE
        ${RKNN_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/3rd_party/rk_platform/mpp/include
        ${CMAKE_SOURCE_DIR}/3rd_party/rk_platform/rga/include)
    # librknnrt.so / librockchip_mpp.so / librga.so 均为 aarch64 库，x86_64 主机仅头文件编译
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_link_libraries(ai_stream_hal PRIVATE ${RKNN_LIBRARIES})
    else()
        target_link_libraries(ai_stream_hal PRIVATE dl)
    endif()
endif()

if(WITH_ASCEND)
    target_compile_definitions(ai_stream_hal PRIVATE WITH_ASCEND ENABLE_DVPP_INTERFACE)
    target_include_directories(ai_stream_hal PRIVATE ${Ascend_INCLUDE_DIRS})
    target_link_libraries(ai_stream_hal PUBLIC ${Ascend_LIBRARIES})
    # CANN 的 libascend_dump/libdvpp_op_base 依赖 driver 的 drv*/hal* 符号，
    # 官方 sample 通过 runtime/lib64/stub 解决；本环境无 stub，运行时由
    # set_env.sh/LD_LIBRARY_PATH 提供 driver 库，链接期允许 shlib 未定义符号。
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        # --allow-shlib-undefined：CANN 内部库的 driver 符号在链接期未解析，
        #   运行时由 CANN driver 提供。
        # --disable-new-dtags：生成 DT_RPATH（可传递），使 libascendcl 的
        #   依赖 libmsprofiler/libascend_dump 等无需 source set_env.sh 也能找到。
        target_link_options(ai_stream_hal INTERFACE
            -Wl,--allow-shlib-undefined
            -Wl,--disable-new-dtags)
    endif()
    # CANN 库不在系统默认路径：把 lib64 与 driver 运行库目录加入 RPATH，
    # 这样不 source set_env.sh 也能启动（与 OpenVINO 的处理一致）。
    if(Ascend_ACL_LIBRARY)
        get_filename_component(_ASCEND_LIB_DIR "${Ascend_ACL_LIBRARY}" DIRECTORY)
        set_property(TARGET ai_stream_hal APPEND PROPERTY BUILD_RPATH "${_ASCEND_LIB_DIR}")
        set_property(TARGET ai_stream_hal APPEND PROPERTY INSTALL_RPATH "${_ASCEND_LIB_DIR}")
    endif()
    foreach(_ascend_driver_dir
            "/usr/local/Ascend/driver/lib64"
            "/usr/local/Ascend/driver/lib64/common"
            "/usr/local/Ascend/driver/lib64/driver")
        if(EXISTS "${_ascend_driver_dir}")
            set_property(TARGET ai_stream_hal APPEND PROPERTY BUILD_RPATH "${_ascend_driver_dir}")
            set_property(TARGET ai_stream_hal APPEND PROPERTY INSTALL_RPATH "${_ascend_driver_dir}")
        endif()
    endforeach()
endif()

if(WITH_HORIZON)
    target_compile_definitions(ai_stream_hal PRIVATE WITH_HORIZON)
    target_include_directories(ai_stream_hal PRIVATE ${HORIZON_DNN_INCLUDE_DIRS})
    # Horizon uses dlopen for libdnn.so, only need dl on non-aarch64 or for linking
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_link_libraries(ai_stream_hal PRIVATE ${HORIZON_DNN_LIBRARIES} dl)
    else()
        target_link_libraries(ai_stream_hal PRIVATE dl)
    endif()
endif()

if(WITH_OPENVINO)
    target_compile_definitions(ai_stream_hal PRIVATE WITH_OPENVINO)
    target_include_directories(ai_stream_hal PRIVATE ${OPENVINO_INCLUDE_DIRS})
    target_link_libraries(ai_stream_hal PRIVATE ${OPENVINO_LIBRARIES})
    # pip wheel / 非系统路径安装时，运行时需要能找到 libopenvino*.so
    if(OPENVINO_LIBRARY_DIR)
        set_target_properties(ai_stream_hal PROPERTIES
            BUILD_RPATH "${OPENVINO_LIBRARY_DIR}"
            INSTALL_RPATH "${OPENVINO_LIBRARY_DIR}"
        )
    endif()
endif()
