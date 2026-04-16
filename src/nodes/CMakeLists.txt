# 基本节点源文件
set(NODES_SOURCES
    registry/node_registry.cpp
    decode/decoder_pool.cpp
    source/rtsp_source.cpp
    decode/ffmpeg_decode.cpp
    draw/osd_draw.cpp
    sink/rtmp_sink.cpp
    sink/mp4_save.cpp
)

# 预处理节点：根据 CUDA 支持选择 CPU 或 GPU 版本
if(WITH_CUDA)
    find_package(CUDA REQUIRED)
    find_package(CUDAToolkit REQUIRED)

    # 使用 GPU 加速的预处理节点
    list(APPEND NODES_SOURCES
        preprocess/gpu_resize_normalize.cu
        postprocess/gpu_detection_post.cu
    )

    # 为 CUDA 文件设置编译选项
    set_source_files_properties(
        preprocess/gpu_resize_normalize.cu
        postprocess/gpu_detection_post.cu
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            COMPILE_FLAGS "-Xcompiler -fPIC"
    )
else()
    # 使用 CPU 版本的预处理节点
    list(APPEND NODES_SOURCES
        preprocess/resize_normalize.cpp
        postprocess/detection_post.cpp
    )
endif()

# 推理节点：条件排除未启用的推理后端
if(WITH_TENSORRT)
    list(APPEND NODES_SOURCES infer/detection_infer.cpp)
endif()

# 创建节点库
add_library(ai_stream_nodes ${NODES_SOURCES})

target_include_directories(ai_stream_nodes
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}           # 包含 src/nodes/
        ${CMAKE_CURRENT_SOURCE_DIR}/registry  # 包含 src/nodes/registry/
)

# Ensure CUDA device code is properly linked
if(WITH_CUDA)
    # Disable separable compilation to avoid device registration issues
    set_source_files_properties(
        preprocess/gpu_resize_normalize.cu
        postprocess/gpu_detection_post.cu
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION OFF
    )
endif()

message("FFMPEG:${FFMPEG_LIBRARIES}")

target_link_libraries(ai_stream_nodes
    PRIVATE 
        ${FFMPEG_LIBRARIES}
        ${FFMPEG_LINK_FLAGS}
        ${CUDA_LIBRARIES}
        ${CUDA_nppc_LIBRARY}
        ${CUDA_nppig_LIBRARY}
        ${CUDA_nppicc_LIBRARY}
        ${CUDA_nppisu_LIBRARY}
        cudadevrt
        cudart
)