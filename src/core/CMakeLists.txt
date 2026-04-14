
add_library(ai_stream_core
    node.cpp
    pipeline_manager.cpp
    thread_pool.cpp
    ${PROJECT_SOURCE_DIR}/3rd_party/log_mgr/log_mgr.cpp
    # frame_queue.h 是 header-only，无需加入源列表
)

target_include_directories(ai_stream_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_SOURCE_DIR}/src          # 这使得可以使用 "nodes/registry/node_factory.h"
        ${PROJECT_SOURCE_DIR}
)