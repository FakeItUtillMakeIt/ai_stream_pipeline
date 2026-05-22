
if(WITH_TRACK)
    add_subdirectory(track)
endif()

# 基本节点源文件
set(NODES_SOURCES
    registry/node_registry.cpp
    decode/decoder_pool.cpp
    source/rtsp_source.cpp
    decode/ffmpeg_decode.cpp
    draw/osd_draw.cpp
    sink/rtmp_sink.cpp
    sink/mp4_save.cpp
    sink/encoder_base.cpp
    preprocess/resize_normalize.cpp
)

# 预处理节点：根据 CUDA 支持选择 CPU 或 GPU 版本
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)

    list(APPEND NODES_SOURCES
        preprocess/gpu_resize_normalize.cu
        postprocess/gpu_detection_post.cu
    )
endif()

# 推理节点
if(WITH_TENSORRT)
    list(APPEND NODES_SOURCES infer/detection_infer.cpp infer/pose_infer.cpp)
endif()

# 动态库
add_library(ai_stream_nodes SHARED ${NODES_SOURCES})

target_include_directories(ai_stream_nodes
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/registry
)

# ==================== CUDA 配置 ====================
if(WITH_CUDA)
    # 必须开启，解决设备链接问题
    set_target_properties(ai_stream_nodes PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
        POSITION_INDEPENDENT_CODE ON
    )

    target_link_libraries(ai_stream_nodes
        PRIVATE
            ${FFMPEG_LIBRARIES}
            # CUDA 运行时 + 设备运行时
            ${CUDA_CUDART_LIBRARY}
            ${CUDA_nppc_LIBRARY}
            ${CUDA_nppig_LIBRARY}
            ${CUDA_nppicc_LIBRARY}
            # TensorRT
            nvinfer
            nvonnxparser
            nvinfer_plugin
    )
else()
    target_link_libraries(ai_stream_nodes
        PRIVATE
            ${FFMPEG_LIBRARIES}
    )
endif()

if (WITH_TRACK)
    target_include_directories(ai_stream_nodes
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/track
            ${PROJECT_SOURCE_DIR}/3rd_party/tracker
            )
            
    target_sources(ai_stream_nodes
        PRIVATE 
            $<TARGET_OBJECTS:ai_stream_track>
    )
    target_link_libraries(ai_stream_nodes
        PUBLIC 
            ai_stream_core
        PRIVATE 
            ocsort
            bytetrack
            Eigen3::Eigen
    )
endif()

if(WITH_ALERT)
    target_include_directories(ai_stream_nodes
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/alert
    )

    target_sources(ai_stream_nodes
        PRIVATE 
            $<TARGET_OBJECTS:alert_node>
    )

    target_link_libraries(ai_stream_nodes
        
        PRIVATE 
            nlohmann_json::nlohmann_json
    )
endif()