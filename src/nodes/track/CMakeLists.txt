# src/nodes/track/CMakeLists.txt

# ============================================================
# 收集源文件
# ============================================================
set(TRACK_SOURCES
    tracker_node.cpp
    ocsort_adapter.cpp
    bytetrack_adapter.cpp
)

set(TRACK_HEADERS
    base_tracker.h
    track_result.h
    tracker_factory.h
    tracker_node.h
    ocsort_adapter.h
    bytetrack_adapter.h
)

# ============================================================
# 创建跟踪节点库
# ============================================================
add_library(ai_stream_track OBJECT ${TRACK_SOURCES} ${TRACK_HEADERS})

target_compile_options(ai_stream_track PRIVATE -fPIC)
# ============================================================
# 包含目录
# ============================================================
target_include_directories(ai_stream_track
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/src
        ${CMAKE_SOURCE_DIR}/src/nodes
        ${PROJECT_SOURCE_DIR}/3rd_party/tracker
        ${PROJECT_SOURCE_DIR}
)

# 在 target_include_directories 之后添加
message(STATUS "=== Track Node Include Directories ===")
get_target_property(TRACK_INCLUDES ai_stream_track INCLUDE_DIRECTORIES)
message(STATUS "INCLUDE_DIRECTORIES: ${TRACK_INCLUDES}")

message(STATUS "=== Project Paths ===")
message(STATUS "CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")

# ============================================================
# 链接依赖
# ============================================================
target_link_libraries(ai_stream_track
    PUBLIC
        ai_stream_core
    PRIVATE
        # ocsort
        bytetrack
        Eigen3::Eigen
        opencv_core
        opencv_imgproc
)

target_compile_features(ai_stream_track PRIVATE cxx_std_17)
