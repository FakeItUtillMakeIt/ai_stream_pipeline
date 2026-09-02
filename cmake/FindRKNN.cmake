# cmake/FindRKNN.cmake
# 查找 Rockchip RKNN SDK
#
# 定义以下变量：
#   RKNN_FOUND         - 是否找到 RKNN SDK
#   RKNN_INCLUDE_DIRS  - 头文件目录
#   RKNN_LIBRARIES     - 库文件
#
# 以下变量可设置以指定路径：
#   RKNN_ROOT          - SDK 安装根目录

# 交叉编译（CMAKE_FIND_ROOT_PATH_MODE=ONLY）时 find_path/find_library 的绝对
# 候选路径会被 reroot 而失效，因此内置库先做确定性存在性检查。
set(_RKNN_VENDORED_INC  "${CMAKE_SOURCE_DIR}/3rd_party/rk_platform/rknn/include")
set(_RKNN_VENDORED_LIB  "${CMAKE_SOURCE_DIR}/3rd_party/rk_platform/rknn/lib/aarch64/librknnrt.so")
if(NOT RKNN_INCLUDE_DIR AND EXISTS "${_RKNN_VENDORED_INC}/rknn_api.h")
    set(RKNN_INCLUDE_DIR "${_RKNN_VENDORED_INC}")
endif()
if(NOT RKNN_LIBRARY AND EXISTS "${_RKNN_VENDORED_LIB}")
    set(RKNN_LIBRARY "${_RKNN_VENDORED_LIB}")
endif()

find_path(RKNN_INCLUDE_DIR
    NAMES rknn_api.h
    PATHS
        ${RKNN_ROOT}/include
        ${CMAKE_SOURCE_DIR}/3rd_party/rk_platform/rknn/include
        ${CMAKE_SOURCE_DIR}/3rd_party/rknn/include
        /usr/include
        /usr/local/include
        /opt/rknn/include
        $ENV{RKNN_ROOT}/include
)

find_library(RKNN_LIBRARY
    NAMES rknnrt rknn_api
    PATHS
        ${RKNN_ROOT}/lib
        ${CMAKE_SOURCE_DIR}/3rd_party/rk_platform/rknn/lib/aarch64
        ${CMAKE_SOURCE_DIR}/3rd_party/rknn/lib/aarch64
        ${CMAKE_SOURCE_DIR}/3rd_party/rk_platform/rknn/lib
        ${CMAKE_SOURCE_DIR}/3rd_party/rknn/lib
        /usr/lib
        /usr/local/lib
        /opt/rknn/lib
        $ENV{RKNN_ROOT}/lib
)

include(FindPackageHandleStandardArgs)

# find_path/find_library 在 ONLY 模式下对源码树绝对路径 reroot 失败时，
# 会把普通变量覆盖为 NOTFOUND——这里做最终恢复，保证内置 SDK 始终可用。
if(NOT RKNN_INCLUDE_DIR AND EXISTS "${_RKNN_VENDORED_INC}/rknn_api.h")
    set(RKNN_INCLUDE_DIR "${_RKNN_VENDORED_INC}")
endif()
if(NOT RKNN_LIBRARY AND EXISTS "${_RKNN_VENDORED_LIB}")
    set(RKNN_LIBRARY "${_RKNN_VENDORED_LIB}")
endif()

find_package_handle_standard_args(RKNN
    REQUIRED_VARS RKNN_LIBRARY RKNN_INCLUDE_DIR
    FAIL_MESSAGE "RKNN SDK not found. Set RKNN_ROOT or install RKNN SDK."
)

if(RKNN_FOUND)
    set(RKNN_INCLUDE_DIRS ${RKNN_INCLUDE_DIR})
    set(RKNN_LIBRARIES ${RKNN_LIBRARY})

    # 从 rknn_api.h 提取 SDK 版本（RKNN_API_VERSION，如 "1.6.0"）
    set(RKNN_VERSION "")
    if(EXISTS "${RKNN_INCLUDE_DIR}/rknn_api.h")
        file(STRINGS "${RKNN_INCLUDE_DIR}/rknn_api.h" _RKNN_VER_LINE
             REGEX "#define[ \t]+RKNN_API_VERSION")
        if(_RKNN_VER_LINE MATCHES "RKNN_API_VERSION[ \t]+\"?([0-9.]+)\"?")
            set(RKNN_VERSION "${CMAKE_MATCH_1}")
        endif()
    endif()
    if(RKNN_VERSION)
        message(STATUS "Found RKNN SDK version: ${RKNN_VERSION}")
    endif()

    # 平台一致性提示：SDK 架构需与目标平台匹配（x86_64/aarch64）
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        message(STATUS "Target platform: aarch64 (请确认 RKNN 库为 aarch64 版本)")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        if(NOT RKNN_ALLOW_X86_SIMULATOR)
            message(FATAL_ERROR "RKNN on x86_64 is simulator-only; set RKNN_ALLOW_X86_SIMULATOR=ON explicitly")
        endif()
        message(WARNING "Target platform: x86_64; RKNN backend is allowed only for simulator use")
    endif()
endif()

mark_as_advanced(RKNN_INCLUDE_DIR RKNN_LIBRARY)
