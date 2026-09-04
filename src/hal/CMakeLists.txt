# src/hal/CMakeLists.txt
# 硬件抽象层（HAL）——按能力分目录：infer（推理）/ encode（编码）/
# decode（解码）/ image_accel（图像加速）/ common（共享工具），
# 能力目录内按厂商分子目录或前缀；根目录为各能力的工厂实现。

set(HAL_SOURCES
    inference_engine_factory.cpp
    detection_inference_engine_factory.cpp
    image_accelerator_factory.cpp
    video_codec_factory.cpp
    action_recognition_factory.cpp
    backend_diagnostics.cpp
)

# CPU fallback 始终编译
list(APPEND HAL_SOURCES
    image_accel/cpu/cpu_image_accelerator.cpp
    infer/cpu/cpu_inference_engine.cpp
    decode/cpu/ffmpeg_video_codec.cpp
    pose_estimation_factory.cpp
)

# TensorRT 后端
if(WITH_TENSORRT)
    find_package(TensorRT REQUIRED)
    list(APPEND HAL_SOURCES
        infer/nvidia/trt_core.cpp
        infer/nvidia/tensorrt_inference_engine.cpp
        infer/nvidia/tensorrt_detection_engine.cpp
        infer/nvidia/tensorrt_action_recognition.cpp
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
        decode/nvidia/nvdec_video_codec.cpp
    )
endif()

# RKNN 后端
if(WITH_RKNN)
    find_package(RKNN REQUIRED)
    list(APPEND HAL_SOURCES
        infer/rk/rknn_inference_engine.cpp
        infer/rk/rknn_detection_engine.cpp
        infer/rk/rknn_pose_estimation.cpp
        image_accel/rk/rga_image_accelerator.cpp
        decode/rk/mpp_video_codec.cpp
        infer/rk/rknn_action_recognition.cpp
    )
endif()

# Ascend 后端
if(WITH_ASCEND)
    find_package(Ascend REQUIRED)
    list(APPEND HAL_SOURCES
        image_accel/ascend/dvpp_image_accelerator.cpp
        decode/ascend/dvpp_video_codec.cpp
        infer/ascend/ascend_action_recognition.cpp
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
    target_compile_definitions(ai_stream_hal PRIVATE WITH_ASCEND)
    target_include_directories(ai_stream_hal PRIVATE ${Ascend_INCLUDE_DIRS})
    target_link_libraries(ai_stream_hal PRIVATE ${Ascend_LIBRARIES})
endif()
