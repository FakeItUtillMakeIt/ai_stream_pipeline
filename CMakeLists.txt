# ==============================================================================
# ai_stream_pipeline - Top-level CMakeLists.txt
# ==============================================================================
cmake_minimum_required(VERSION 3.16)
project(ai_stream_pipeline
    VERSION 1.0.0
    DESCRIPTION "Modular AI-driven video stream processing pipeline"
    LANGUAGES CXX
)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "ai_stream_pipeline currently supports Linux only; use a Linux target or add a platform port first")
endif()

# ------------------------------------------------------------------------------
# 1. 选项配置
# ------------------------------------------------------------------------------
set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build, options are: Debug Release MinSizeRel RelWithDebInfo.")
option(BUILD_HTTP_SERVER "Build HTTP API server executable" ON)
option(BUILD_TESTS       "Build unit and integration tests" OFF)
option(BUILD_EXAMPLES    "Build example applications" ON)
option(BUILD_TOOLS       "Build development tools (benchmark)" ON)
option(ENABLE_COVERAGE   "Enable code coverage instrumentation" OFF)
option(INSTALL_HEADERS   "Install public headers" ON)

# 推理后端选项
# 注：ONNX Runtime 支持（原 WITH_ONNX）已移除——环境中无 ONNXRuntime 且无对应节点实现，
# 如需引入请同时提供 onnx_infer 节点代码与依赖查找模块
option(WITH_TENSORRT     "Enable TensorRT inference support" OFF)

# GPU 加速选项
option(WITH_CUDA "Enable CUDA support for GPU acceleration" OFF)
option(WITH_NPP "Enable NPP library for GPU image processing" OFF)

# 新平台推理后端
option(WITH_RKNN  "Enable RKNN inference support (RK3588)" OFF)
option(WITH_ASCEND "Enable Ascend CANN inference support" OFF)
option(RKNN_ALLOW_X86_SIMULATOR "Allow RKNN backend on x86_64 for simulator-only builds" OFF)

# CPU fallback：无 GPU 时用 CPU 实现预处理/后处理/绘制
option(WITH_CPU_FALLBACK "Enable CPU fallback for preprocess/postprocess/draw" ON)

option(WITH_TRACK "Enable tracking node" ON)
option(WITH_ALERT "Enable alert node" ON)

# ------------------------------------------------------------------------------
# 2. 全局编译设置
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# CUDA 语言按需启用
if(WITH_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    set(CMAKE_CUDA_RESOLVE_DEVICE_SYMBOLS ON)
endif()

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

# 安装目录变量（需在所有 add_subdirectory 之前定义，供子目录 install 规则使用）
include(GNUInstallDirs)

# ------------------------------------------------------------------------------
# 3. 依赖管理
# ------------------------------------------------------------------------------
include(cmake/dependencies.cmake)

# 查找或获取基础依赖（spdlog, nlohmann_json）
# 优先使用系统 spdlog，找不到时从 GitHub 下载
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    message(STATUS "System spdlog not found, fetching v1.9.2 from GitHub...")
    include(FetchContent)
    FetchContent_Declare(
        spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG v1.9.2
        GIT_SHALLOW TRUE
    )
    set(SPDLOG_BUILD_SHARED ON CACHE BOOL "" FORCE)
    set(SPDLOG_INSTALL OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(spdlog)
endif()
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# 视频/图像处理核心依赖
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio dnn)

# 可选的 OpenCV freetype 模块（用于中文绘制等高级文本渲染）
# 当系统安装了 opencv_freetype 组件时，OpenCVConfig.cmake 会导出 opencv_freetype target
if(TARGET opencv_freetype)
    set(HAVE_OPENCV_FREETYPE TRUE)
    message(STATUS "OpenCV freetype module available")
else()
    set(HAVE_OPENCV_FREETYPE FALSE)
    message(STATUS "OpenCV freetype module not found, text rendering will use cv::putText fallback")
endif()
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
    # 注意：此前此处有 -Wl,--allow-multiple-definition 全局 hack，
    # 用于掩盖 FFmpeg/libgcc 相关的符号重复定义问题，已移除。
    # 若链接报 multiple definition，请定位具体重复符号并修复根因。
    message(STATUS "FFmpeg libgcc workaround removed; link errors will surface root cause")
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
    if(NOT WITH_CUDA)
        message(FATAL_ERROR "WITH_TENSORRT requires WITH_CUDA=ON")
    endif()
    find_package(TensorRT QUIET)
    if(TensorRT_FOUND)
        message(STATUS "Found TensorRT: ${TensorRT_INCLUDE_DIRS}")
    else()
        message(WARNING "TensorRT not found. TensorRT inference node will be disabled.")
        set(WITH_TENSORRT OFF)
    endif()
endif()

if(WITH_CUDA)
    # CUDA 架构：未显式指定时通过 nvidia-smi 探测本机 GPU 计算能力（如 8.6 → 86），
    # 并钳制到当前 nvcc 支持的范围内（如 CUDA 12.6 不支持 compute_120）
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
        find_program(NVIDIA_SMI nvidia-smi)
        set(_GPU_ARCH "")
        if(NVIDIA_SMI)
            execute_process(COMMAND ${NVIDIA_SMI} --query-gpu=compute_cap --format=csv,noheader
                            OUTPUT_VARIABLE _GPU_COMPUTE_CAP
                            OUTPUT_STRIP_TRAILING_WHITESPACE
                            ERROR_QUIET RESULT_VARIABLE _NVSMI_RET)
            if(_NVSMI_RET EQUAL 0 AND _GPU_COMPUTE_CAP MATCHES "([0-9]+)\\.([0-9]+)")
                string(REPLACE "." "" _GPU_ARCH "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
            endif()
        endif()

        # 钳制到 nvcc 支持的最高不超过探测值/默认值的架构
        execute_process(COMMAND ${CMAKE_CUDA_COMPILER} --list-gpu-arch
                        OUTPUT_VARIABLE _NVCC_ARCH_LIST
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                        ERROR_QUIET RESULT_VARIABLE _NVCC_LIST_RET)
        set(_SUPPORTED_ARCHS "")
        if(_NVCC_LIST_RET EQUAL 0)
            string(REPLACE "\n" ";" _NVCC_ARCH_LINES "${_NVCC_ARCH_LIST}")
            foreach(_line ${_NVCC_ARCH_LINES})
                if(_line MATCHES "compute_([0-9]+)")
                    list(APPEND _SUPPORTED_ARCHS "${CMAKE_MATCH_1}")
                endif()
            endforeach()
        endif()

        set(_TARGET_ARCH "")
        if(NOT _GPU_ARCH STREQUAL "")
            set(_TARGET_ARCH "${_GPU_ARCH}")
            message(STATUS "Detected GPU compute capability ${_GPU_ARCH}")
        else()
            set(_TARGET_ARCH "80")
            message(STATUS "No GPU detected, defaulting to CUDA arch 80")
        endif()

        if(_SUPPORTED_ARCHS)
            list(SORT _SUPPORTED_ARCHS)
            list(GET _SUPPORTED_ARCHS -1 _MAX_SUPPORTED)
            if(_GPU_ARCH GREATER _MAX_SUPPORTED)
                message(WARNING
                    "Local GPU compute capability ${_GPU_ARCH} exceeds max architecture "
                    "${_MAX_SUPPORTED} supported by ${CMAKE_CUDA_COMPILER} "
                    "(CUDA ${CMAKE_CUDA_COMPILER_VERSION}). Building for ${_MAX_SUPPORTED}; "
                    "the binary may not run on this GPU. Upgrade CUDA toolkit to support it.")
                set(_TARGET_ARCH "${_MAX_SUPPORTED}")
            endif()
        endif()

        set(CMAKE_CUDA_ARCHITECTURES "${_TARGET_ARCH}" CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "Using CUDA arch: ${CMAKE_CUDA_ARCHITECTURES}")
    endif()
endif()

# RKNN 后端（RK3588）
if(WITH_RKNN)
    find_package(RKNN QUIET)
    if(RKNN_FOUND)
        message(STATUS "Found RKNN: ${RKNN_INCLUDE_DIRS}")
    else()
        message(WARNING "RKNN not found. RKNN inference node will be disabled.")
        set(WITH_RKNN OFF)
    endif()
endif()

# Ascend CANN 后端
if(WITH_ASCEND)
    find_package(Ascend QUIET)
    if(Ascend_FOUND)
        message(STATUS "Found Ascend CANN: ${Ascend_INCLUDE_DIRS}")
    else()
        message(WARNING "Ascend CANN not found. Ascend inference node will be disabled.")
        set(WITH_ASCEND OFF)
    endif()
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
# 4.5 硬件抽象层（HAL）
# ------------------------------------------------------------------------------
add_subdirectory(src/hal)

# ------------------------------------------------------------------------------
# 5. 节点库构建（插件）
# ------------------------------------------------------------------------------
add_subdirectory(src/nodes)


target_link_libraries(ai_stream_nodes
    PUBLIC
        ai_stream_core
        ai_stream_hal
    PRIVATE
        opencv_core
        opencv_imgproc
        opencv_videoio
        FFmpeg::FFmpeg
)

# 条件链接 OpenCV freetype 模块
if(HAVE_OPENCV_FREETYPE)
    target_link_libraries(ai_stream_nodes PRIVATE opencv_freetype)
    target_compile_definitions(ai_stream_nodes PRIVATE HAVE_OPENCV_FREETYPE)
endif()

# 条件链接推理后端
if(WITH_TENSORRT)
    target_compile_definitions(ai_stream_nodes PRIVATE WITH_TENSORRT)
    target_include_directories(ai_stream_nodes PRIVATE ${TensorRT_INCLUDE_DIRS})
    target_link_libraries(ai_stream_nodes PRIVATE ${TensorRT_LIBRARIES} nvinfer_plugin)
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
# 8.5 开发工具
# ------------------------------------------------------------------------------
if(BUILD_TOOLS)
    add_subdirectory(tools/benchmark)
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
    # 公开头文件（node.h / i_alert_rule.h）依赖 utils/ 下的 header-only 工具
    install(DIRECTORY utils/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/utils
        COMPONENT devel
        FILES_MATCHING PATTERN "*.h"
    )
endif()

# 安装核心库和节点库（跟踪库仅在启用时存在，需一并导出以满足 PRIVATE 链接依赖）
set(_INSTALL_EXPORT_TARGETS ai_stream_core ai_stream_nodes ai_stream_hal)
if(WITH_TRACK)
    list(APPEND _INSTALL_EXPORT_TARGETS ocsort bytetrack)
endif()


install(TARGETS ${_INSTALL_EXPORT_TARGETS}
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

# 安装 CMake 导出文件
install(EXPORT ai_stream_pipeline-targets
    FILE ai_stream_pipelineTargets.cmake
    NAMESPACE ai_stream_pipeline::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai_stream_pipeline
)

# 生成并安装版本文件与包配置文件
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/ai_stream_pipelineConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ai_stream_pipelineConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/ai_stream_pipelineConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai_stream_pipeline
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/ai_stream_pipelineConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/ai_stream_pipelineConfigVersion.cmake"
    # 自定义 Find 模块随包安装，供 Config 的 find_dependency 使用
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindFFmpeg.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindTensorRT.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindRKNN.cmake"
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/FindAscend.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ai_stream_pipeline
)

# 安装运行时配置文件目录
install(DIRECTORY config/
    DESTINATION ${CMAKE_INSTALL_DATADIR}/ai_stream_pipeline/config
    COMPONENT runtime
)

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
message(STATUS "  CUDA support:         ${WITH_CUDA}")
message(STATUS "  TensorRT support:     ${WITH_TENSORRT}")
message(STATUS "  RKNN support:         ${WITH_RKNN}")
message(STATUS "  Ascend support:       ${WITH_ASCEND}")
message(STATUS "  CPU fallback:         ${WITH_CPU_FALLBACK}")
message(STATUS "  WITH_FTP:             ${WITH_FTP}")
message(STATUS "  Install prefix:       ${CMAKE_INSTALL_PREFIX}")
message(STATUS "===============================================")
