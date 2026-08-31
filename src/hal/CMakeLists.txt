# src/hal/CMakeLists.txt
# 硬件抽象层（HAL）——推理引擎、图像加速、视频编解码的后端抽象

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
    cpu/cpu_image_accelerator.cpp
    cpu/cpu_inference_engine.cpp
    ffmpeg/ffmpeg_video_codec.cpp
    pose_estimation_factory.cpp
)

# TensorRT 后端
if(WITH_TENSORRT)
    find_package(TensorRT REQUIRED)
    list(APPEND HAL_SOURCES
        tensorrt/trt_core.cpp
        tensorrt/tensorrt_inference_engine.cpp
        tensorrt/tensorrt_detection_engine.cpp
        tensorrt/tensorrt_action_recognition.cpp
        tensorrt/tensorrt_pose_estimation.cpp
    )
    # 姿态估计 GPU 预处理 kernel（需 CUDA 语言）
    if(WITH_CUDA)
        list(APPEND HAL_SOURCES tensorrt/tensorrt_pose_kernels.cu)
    endif()
endif()

if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
    list(APPEND HAL_SOURCES
        npp/npp_image_accelerator.cpp
        npp/cuda_kernels_wrapper.cu
        nvdec/nvdec_video_codec.cpp
    )
endif()

# RKNN 后端
if(WITH_RKNN)
    find_package(RKNN REQUIRED)
    list(APPEND HAL_SOURCES
        rknn/rknn_inference_engine.cpp
        rga/rga_image_accelerator.cpp
        mpp/mpp_video_codec.cpp
        rknn/rknn_action_recognition.cpp
    )
endif()

# Ascend 后端
if(WITH_ASCEND)
    find_package(Ascend REQUIRED)
    list(APPEND HAL_SOURCES
        ascend/ascend_inference_engine.cpp
        ascend/dvpp_image_accelerator.cpp
        ascend/dvpp_video_codec.cpp
        ascend/ascend_action_recognition.cpp
    )
endif()

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
    target_include_directories(ai_stream_hal PRIVATE ${RKNN_INCLUDE_DIRS})
    target_link_libraries(ai_stream_hal PRIVATE ${RKNN_LIBRARIES})
endif()

if(WITH_ASCEND)
    target_compile_definitions(ai_stream_hal PRIVATE WITH_ASCEND)
    target_include_directories(ai_stream_hal PRIVATE ${Ascend_INCLUDE_DIRS})
    target_link_libraries(ai_stream_hal PRIVATE ${Ascend_LIBRARIES})
endif()
