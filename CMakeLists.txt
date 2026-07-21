# ==============================================================================
# ai_stream_pipeline - Top-level CMakeLists.txt
# ==============================================================================
cmake_minimum_required(VERSION 3.16)
project(ai_stream_pipeline
    VERSION 1.0.0
    DESCRIPTION "Modular AI-driven video stream processing pipeline"
    LANGUAGES CXX CUDA
)

# ------------------------------------------------------------------------------
# 1. 选项配置
# ------------------------------------------------------------------------------
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release MinSizeRel RelWithDebInfo.")
option(BUILD_HTTP_SERVER "Build HTTP API server executable" ON)
option(BUILD_TESTS       "Build unit and integration tests" OFF)
option(BUILD_EXAMPLES    "Build example applications" ON)
option(ENABLE_COVERAGE   "Enable code coverage instrumentation" OFF)
option(INSTALL_HEADERS   "Install public headers" ON)

# 推理后端选项（互斥或可选）
option(WITH_TENSORRT     "Enable TensorRT inference support" ON)
option(WITH_ONNX         "Enable ONNX Runtime inference support" OFF)

# GPU 加速选项
option(WITH_CUDA "Enable CUDA support for GPU acceleration" ON)
option(WITH_NPP "Enable NPP library for GPU image processing" ON)

option(WITH_TRACK "Enable tracking node" ON)
option(WITH_ALERT "Enable alert node" ON)

# ------------------------------------------------------------------------------
# 2. 全局编译设置
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# 导出 compile_commands.json（用于 IDE 和 clang-tidy）
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 默认构建类型为 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# 覆盖率编译标志
if(ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fprofile-arcs -ftest-coverage")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fprofile-arcs -ftest-coverage")
    endif()
endif()

# 将 cmake/ 目录加入模块搜索路径
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# ------------------------------------------------------------------------------
# 3. 依赖管理
# ------------------------------------------------------------------------------
include(cmake/dependencies.cmake)

# 查找或获取基础依赖（spdlog, nlohmann_json）
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# 视频/图像处理核心依赖
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio)
find_package(FFmpeg REQUIRED COMPONENTS avcodec avformat avutil swscale)

# FTP上传依赖（可选）
find_package(CURL QUIET)
if(CURL_FOUND)
    message(STATUS "Found CURL: ${CURL_LIBRARIES}")
    set(WITH_FTP ON)
else()
    message(WARNING "CURL not found, FTP upload feature will be disabled")
    set(WITH_FTP OFF)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # 方法1：允许重复定义（快速修复）
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,--allow-multiple-definition")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--allow-multiple-definition")
    
    
    message(STATUS "Applied FFmpeg libgcc conflict workaround")
endif()

# HTTP 服务依赖（如果构建 HTTP 服务器）
if(BUILD_HTTP_SERVER)
    include(FetchContent)
    FetchContent_Declare(
        httplib
        GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
        GIT_TAG        v0.14.3   # 或者使用最新的稳定版本
    )
    FetchContent_MakeAvailable(httplib)
    # 注意：FetchContent 之后，httplib 的目标名称通常是 httplib::httplib
endif()

# 推理后端可选依赖
if(WITH_TENSORRT)
    find_package(TensorRT QUIET)
    if(TensorRT_FOUND)
        message(STATUS "Found TensorRT: ${TensorRT_INCLUDE_DIRS}")
    else()
        message(WARNING "TensorRT not found. TensorRT inference node will be disabled.")
        set(WITH_TENSORRT OFF)
    endif()
endif()

if(WITH_ONNX)
    find_package(ONNXRuntime QUIET)
    if(NOT ONNXRuntime_FOUND)
        message(WARNING "ONNX Runtime not found. ONNX inference node will be disabled.")
        set(WITH_ONNX OFF)
    endif()
endif()

if(WITH_CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA architectures")
endif()

if(WITH_TRACK)
    find_package(Eigen3 REQUIRED)
    if(EIGEN3_FOUND)
        message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
    else(EIGEN3_FOUND)
        message(WARNING "Eigen3 not found. Tracking node will be disabled.")
        set(WITH_TRACK OFF)
    endif()
    add_subdirectory(${PROJECT_SOURCE_DIR}/3rd_party/tracker)
endif()

if(WITH_ALERT)
    add_subdirectory(src/rules/alert)
    add_subdirectory(src/nodes/alert)
endif()

# 测试依赖
if(BUILD_TESTS)
    find_package(GTest REQUIRED)
    include(GoogleTest)
endif()

# ------------------------------------------------------------------------------
# 4. 核心库构建
# ------------------------------------------------------------------------------
add_subdirectory(src/core)

# 公开头文件目录
target_include_directories(ai_stream_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${OpenCV_INCLUDE_DIRS}
)

# 核心库链接基础依赖
target_link_libraries(ai_stream_core
    PUBLIC
        spdlog::spdlog
        nlohmann_json::nlohmann_json
    PRIVATE
        opencv_core
        opencv_imgproc
)

# 设置核心库属性
set_target_properties(ai_stream_core PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION 1
    EXPORT_NAME core
)

# ------------------------------------------------------------------------------
# 5. 节点库构建（插件）
# ------------------------------------------------------------------------------
add_subdirectory(src/nodes)


target_link_libraries(ai_stream_nodes
    PUBLIC
        ai_stream_core
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_videoio
        opencv_freetype
        ${FFMPEG_LIBRARIES}
)

# 条件链接推理后端
if(WITH_TENSORRT)
    target_compile_definitions(ai_stream_nodes PRIVATE WITH_TENSORRT)
    target_include_directories(ai_stream_nodes PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(ai_stream_nodes PRIVATE ${TensorRT_LIBRARIES})
endif()

if(WITH_ONNX)
    target_compile_definitions(ai_stream_nodes PRIVATE WITH_ONNX)
    target_include_directories(ai_stream_nodes PRIVATE ${ONNXRuntime_INCLUDE_DIRS})
    target_link_libraries(ai_stream_nodes PRIVATE ${ONNXRuntime_LIBRARIES})
endif()

# ------------------------------------------------------------------------------
# 6. HTTP 服务器可执行文件
# ------------------------------------------------------------------------------
if(BUILD_HTTP_SERVER)
    add_subdirectory(src/http)
    
    target_link_libraries(http_server
        PRIVATE
            ai_stream_core
            ai_stream_nodes
            httplib::httplib
    )
    
    # 如果节点库使用了 whole-archive，这里要关闭
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_link_options(http_server PRIVATE -Wl,--no-whole-archive)
    endif()
endif()

# ------------------------------------------------------------------------------
# 7. 测试
# ------------------------------------------------------------------------------
if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ------------------------------------------------------------------------------
# 8. 示例
# ------------------------------------------------------------------------------
if(BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

# ------------------------------------------------------------------------------
# 9. 安装规则
# ------------------------------------------------------------------------------
include(GNUInstallDirs)

# 安装公开头文件
if(INSTALL_HEADERS)
    install(DIRECTORY include/ai_stream
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        COMPONENT devel
    )
endif()

# 安装核心库和节点库
install(TARGETS ai_stream_core ai_stream_nodes
    # ocsort 
    bytetrack
    EXPORT ai_stream_pipeline-targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# 安装 HTTP 服务器
if(BUILD_HTTP_SERVER)
    install(TARGETS http_server
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    )
endif()

# # 安装 CMake 导出文件
# install(EXPORT ai_stream_pipeline-targets
#     FILE ai_stream_pipelineTargets.cmake
#     NAMESPACE ai_stream_pipeline::
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai_stream_pipeline
# )

# # 生成并安装版本文件
# include(CMakePackageConfigHelpers)
# write_basic_package_version_file(
#     "${CMAKE_CURRENT_BINARY_DIR}/ai_stream_pipelineConfigVersion.cmake"
#     VERSION ${PROJECT_VERSION}
#     COMPATIBILITY SameMajorVersion
# )

# configure_package_config_file(
#     "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ai_stream_pipelineConfig.cmake.in"
#     "${CMAKE_CURRENT_BINARY_DIR}/ai_stream_pipelineConfig.cmake"
#     INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai_stream_pipeline
# )

# install(FILES
#     "${CMAKE_CURRENT_BINARY_DIR}/ai_stream_pipelineConfig.cmake"
#     "${CMAKE_CURRENT_BINARY_DIR}/ai_stream_pipelineConfigVersion.cmake"
#     DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai_stream_pipeline
# )

# # 安装配置文件目录（可选）
# install(DIRECTORY config/
#     DESTINATION ${CMAKE_INSTALL_DATADIR}/ai_stream_pipeline/config
#     COMPONENT runtime
# )

# ------------------------------------------------------------------------------
# 10. 输出构建摘要
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== ai_stream_pipeline Configuration Summary ===")
message(STATUS "  Build type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Standard:         ${CMAKE_CXX_STANDARD}")
message(STATUS "  HTTP Server:          ${BUILD_HTTP_SERVER}")
message(STATUS "  Build tests:          ${BUILD_TESTS}")
message(STATUS "  Build examples:       ${BUILD_EXAMPLES}")
message(STATUS "  TensorRT support:     ${WITH_TENSORRT}")
message(STATUS "  ONNX Runtime support: ${WITH_ONNX}")
message(STATUS "  Install prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===============================================")