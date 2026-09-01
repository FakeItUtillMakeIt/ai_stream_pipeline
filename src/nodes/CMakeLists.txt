
if(WITH_TRACK)
    add_subdirectory(track)
endif()

# 基本节点源文件
set(NODES_SOURCES
    decode/decoder_pool.cpp
    source/rtsp_source.cpp
    source/file_source.cpp
    decode/ffmpeg_decode.cpp
    draw/osd_draw.cpp
    draw/draw_panel.cpp
    sink/rtmp_sink.cpp
    sink/mp4_save.cpp
    sink/encoder_base.cpp
    preprocess/resize_normalize.cpp
    postprocess/detection_post.cpp
    fusion/fusion_node.cpp
)

# CPU fallback 节点（无 GPU 时使用）
if(WITH_CPU_FALLBACK)
    list(APPEND NODES_SOURCES
        # CPU 版本的预处理/后处理已在上面注册
        # 这里可以添加更多 CPU 专用实现
    )
endif()

# 预处理节点：根据 CUDA 支持选择 CPU 或 GPU 版本
if(WITH_CUDA)
    find_package(CUDAToolkit REQUIRED)
endif()

# 统一推理节点（HAL 抽象，自动选择后端）
# detection 需要 GPU/CUDA（TRT）或 RKNN/Ascend；pose/videomae 可 CPU
if(WITH_CUDA OR WITH_TENSORRT OR WITH_RKNN OR WITH_ASCEND)
    list(APPEND NODES_SOURCES infer/detection_infer.cpp)
endif()
list(APPEND NODES_SOURCES
    infer/pose_infer.cpp
    infer/action_recognition_videomae.cpp
)
# INT8 校准仅 TensorRT 需要
if(WITH_TENSORRT)
    list(APPEND NODES_SOURCES infer/int8_calibrator.cpp)
endif()

# 动态库
add_library(ai_stream_nodes SHARED ${NODES_SOURCES})
set_target_properties(ai_stream_nodes PROPERTIES EXPORT_NAME nodes)

# 编译定义
if(WITH_TENSORRT)
    target_compile_definitions(ai_stream_nodes PRIVATE WITH_TENSORRT)
endif()

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

    target_compile_definitions(ai_stream_nodes PRIVATE WITH_CUDA)

    target_link_libraries(ai_stream_nodes
        PRIVATE
            FFmpeg::FFmpeg
            CUDA::cudart
            CUDA::nppc
            CUDA::nppig
            CUDA::nppicc
            CUDA::cublas
            CUDA::cusolver
    )
else()
    target_link_libraries(ai_stream_nodes
        PRIVATE
            FFmpeg::FFmpeg
    )
endif()

# ==================== RKNN 配置 ====================
if(WITH_RKNN)
    target_compile_definitions(ai_stream_nodes PRIVATE WITH_RKNN)
    target_include_directories(ai_stream_nodes PRIVATE
        ${RKNN_INCLUDE_DIRS}
        ${CMAKE_SOURCE_DIR}/3rd_party/rk_series/mpp/include
        ${CMAKE_SOURCE_DIR}/3rd_party/rk_series/rga/include)
    # librknnrt.so 是 aarch64 库，x86_64 编译主机不链接（dlopen 动态加载）
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        target_link_libraries(ai_stream_nodes PRIVATE ${RKNN_LIBRARIES})
    else()
        target_link_libraries(ai_stream_nodes PRIVATE dl)
    endif()
endif()

# ==================== Ascend 配置 ====================
if(WITH_ASCEND)
    target_compile_definitions(ai_stream_nodes PRIVATE WITH_ASCEND)
    target_include_directories(ai_stream_nodes PRIVATE ${Ascend_INCLUDE_DIRS})
    target_link_libraries(ai_stream_nodes PRIVATE ${Ascend_LIBRARIES})
endif()

if(WITH_TRACK)
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
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--whole-archive>
            ai_stream_hal
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--no-whole-archive>
            ocsort
            bytetrack
            Eigen3::Eigen
    )
else()
    target_link_libraries(ai_stream_nodes
        PUBLIC
            ai_stream_core
        PRIVATE
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--whole-archive>
            ai_stream_hal
            $<$<CXX_COMPILER_ID:GNU,Clang>:-Wl,--no-whole-archive>
    )
endif()

if(WITH_ALERT)
    target_include_directories(ai_stream_nodes
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/alert
            ${CMAKE_CURRENT_SOURCE_DIR}/../rules/alert
    )

    target_sources(ai_stream_nodes
        PRIVATE
            $<TARGET_OBJECTS:alert_node>
            $<TARGET_OBJECTS:alert_rules>
    )

    target_link_libraries(ai_stream_nodes

        PRIVATE
            nlohmann_json::nlohmann_json
    )
endif()

# Evidence node
add_subdirectory(evidence)
target_include_directories(ai_stream_nodes
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/evidence
)
target_sources(ai_stream_nodes
    PRIVATE
        $<TARGET_OBJECTS:evidence_node>
)
if(WITH_FTP)
    target_link_libraries(ai_stream_nodes
        PRIVATE
            CURL::libcurl
    )
endif()
