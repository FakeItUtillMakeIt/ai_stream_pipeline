# 3rd_party/tracker/CMakeLists.txt

# 确保 Eigen3 已找到
find_package(Eigen3 REQUIRED)

# ============================================================
# OCSort 库（纯头文件，使用 INTERFACE）
# ============================================================
# 收集头文件（可选，用于 IDE 显示）
file(GLOB OCSORT_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/OCSort/*.cpp")

file(GLOB OCSORT_HEADERS 
    "${CMAKE_CURRENT_SOURCE_DIR}/OCSort/*.h" 
    "${CMAKE_CURRENT_SOURCE_DIR}/OCSort/*.hpp")

add_library(ocsort SHARED ${OCSORT_SOURCES} ${OCSORT_HEADERS})

target_include_directories(ocsort
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/OCSort
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(ocsort
    INTERFACE
        Eigen3::Eigen
)

# 添加头文件到目标（仅用于 IDE 显示，不影响编译）
target_sources(ocsort INTERFACE ${OCSORT_HEADERS})

# ============================================================
# ByteTrack 库（有源文件，编译为 SHARED）
# ============================================================
# 收集 ByteTrack 源文件
file(GLOB_RECURSE BYTETRACK_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/ByteTrack/*.cpp"
)

file(GLOB_RECURSE BYTETRACK_HEADERS
    "${CMAKE_CURRENT_SOURCE_DIR}/ByteTrack/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/ByteTrack/*.h"
)

add_library(bytetrack SHARED ${BYTETRACK_SOURCES} ${BYTETRACK_HEADERS})

target_include_directories(bytetrack
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/ByteTrack
        ${CMAKE_CURRENT_SOURCE_DIR}/ByteTrack/common
        ${EIGEN3_INCLUDE_DIR}
)

target_link_libraries(bytetrack
    PUBLIC
        Eigen3::Eigen
        opencv_core
        opencv_imgproc
)

target_compile_features(bytetrack PRIVATE cxx_std_14)

# 设置动态库属性
set_target_properties(bytetrack PROPERTIES
    VERSION 1.0.0
    SOVERSION 1
)