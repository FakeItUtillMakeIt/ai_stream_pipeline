add_library(ai_stream_nodes
    registry/node_registry.cpp
    decode/decoder_pool.cpp
    source/rtsp_source.cpp
    decode/ffmpeg_decode.cpp
    preprocess/resize_normalize.cpp
    infer/tensorrt_infer.cpp
    postprocess/detection_post.cpp
    draw/osd_draw.cpp
    sink/rtmp_sink.cpp
    sink/mp4_save.cpp
)

# 条件排除未启用的推理后端
if(NOT WITH_TENSORRT)
    list(REMOVE_ITEM ai_stream_nodes_SRCS infer/tensorrt_infer.cpp)
endif()

target_include_directories(ai_stream_nodes
    PRIVATE
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}           # 包含 src/nodes/
        ${CMAKE_CURRENT_SOURCE_DIR}/registry  # 包含 src/nodes/registry/
)

message("FFMPEG:${FFMPEG_LIBRARIES}")

target_link_libraries(ai_stream_nodes
    PRIVATE 
        ${FFMPEG_LIBRARIES}
        ${FFMPEG_LINK_FLAGS}
)