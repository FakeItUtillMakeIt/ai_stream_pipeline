# src/hal/CMakeLists.txt
# 硬件抽象层（HAL）——推理引擎、图像加速、视频编解码的后端抽象

set(HAL_SOURCES
    inference_engine_factory.cpp
    image_accelerator_factory.cpp
    video_codec_factory.cpp
    action_recognition_factory.cpp
)

# CPU fallback 始终编译
list(APPEND HAL_SOURCES
    cpu/cpu_image_accelerator.cpp
    cpu/cpu_inference_engine.cpp
    ffmpeg/ffmpeg_video_codec.cpp
)

# TensorRT 后端
if(WITH_TENSORRT)
    find_package(TensorRT REQUIRED)
    list(APPEND HAL_SOURCES
        tensorrt/tensorrt_inference_engine.cpp
        tensorrt/tensorrt_action_recognition.cpp
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

add_library(ai_stream_hal STATIC ${HAL_SOURCES})

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
        avcodec
        avutil
        swscale
)

# 条件链接推理后端
if(WITH_TENSORRT)
    target_compile_definitions(ai_stream_hal PRIVATE WITH_TENSORRT)
    target_include_directories(ai_stream_hal PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(ai_stream_hal PRIVATE ${TensorRT_LIBRARIES})
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
