# cmake/FindAscend.cmake
# 查找华为 Ascend CANN SDK
#
# 定义以下变量：
#   Ascend_FOUND         - 是否找到 CANN SDK
#   Ascend_INCLUDE_DIRS  - 头文件目录
#   Ascend_LIBRARIES     - 库文件（ascendcl，可选 acl_dvpp）
#   Ascend_VERSION       - ACL 版本（如 8.5）
#
# 可设置：
#   ASCEND_HOME           - CANN 安装根目录
#                           （如 /usr/local/Ascend/ascend-toolkit/latest）
#   也自动识别 set_env.sh 导出的 ASCEND_HOME_PATH 环境变量。
#
# CANN 8.x 的实际布局是 <root>/x86_64-linux/{include,lib64}（aarch64 对应
# aarch64-linux），同时 <root>/include、<root>/lib64 通常是同内容的软链接。
# 这里两种布局都搜索。

if(NOT ASCEND_HOME AND DEFINED ENV{ASCEND_HOME})
    set(ASCEND_HOME "$ENV{ASCEND_HOME}")
endif()
if(NOT ASCEND_HOME AND DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    set(_ASCEND_ARCH_DIR "aarch64-linux")
else()
    set(_ASCEND_ARCH_DIR "x86_64-linux")
endif()

set(_ASCEND_ROOTS
    ${ASCEND_HOME}
    /usr/local/Ascend/ascend-toolkit/latest
    /usr/local/Ascend/cann
    /usr/local/Ascend/nnrt/latest
)

set(_ASCEND_INCLUDE_HINTS "")
set(_ASCEND_LIB_HINTS "")
foreach(_root ${_ASCEND_ROOTS})
    if(_root)
        list(APPEND _ASCEND_INCLUDE_HINTS
            "${_root}/include"
            "${_root}/${_ASCEND_ARCH_DIR}/include")
        list(APPEND _ASCEND_LIB_HINTS
            "${_root}/lib64"
            "${_root}/${_ASCEND_ARCH_DIR}/lib64")
    endif()
endforeach()

find_path(Ascend_INCLUDE_DIR
    NAMES acl/acl.h
    HINTS ${_ASCEND_INCLUDE_HINTS}
)

# ACL 主库叫 libascendcl.so（历史版本也出现过 asccl 命名，保留兜底）
find_library(Ascend_ACL_LIBRARY
    NAMES ascendcl asccl
    HINTS ${_ASCEND_LIB_HINTS}
)

find_library(Ascend_ACL_DVPP_LIBRARY
    NAMES acl_dvpp
    HINTS ${_ASCEND_LIB_HINTS}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ascend
    REQUIRED_VARS Ascend_ACL_LIBRARY Ascend_INCLUDE_DIR
    FAIL_MESSAGE "Ascend CANN SDK not found. Set ASCEND_HOME/ASCEND_HOME_PATH or install CANN SDK."
)

if(Ascend_FOUND)
    set(Ascend_INCLUDE_DIRS ${Ascend_INCLUDE_DIR})
    set(Ascend_LIBRARIES ${Ascend_ACL_LIBRARY})
    if(Ascend_ACL_DVPP_LIBRARY)
        list(APPEND Ascend_LIBRARIES ${Ascend_ACL_DVPP_LIBRARY})
    endif()

    # CANN 的 libacl_dvpp.so 在链接期还会引用这些内部库（安全函数/性能/内存/
    # runtime/dlog）。官方 sample 通过 runtime/lib64/stub 解决；本环境没有
    # stub，这里显式补齐，避免可执行文件链接时出现 undefined reference。
    foreach(_ascend_extra acl_dvpp_mpi profapi mmpa runtime unified_dlog c_sec)
        find_library(_ascend_extra_lib_${_ascend_extra}
            NAMES ${_ascend_extra}
            HINTS ${_ASCEND_LIB_HINTS})
        if(_ascend_extra_lib_${_ascend_extra})
            list(APPEND Ascend_LIBRARIES "${_ascend_extra_lib_${_ascend_extra}}")
        endif()
    endforeach()

    # 从 acl_base.h 提取 CANN 版本（ACL_MAJOR_VERSION / ACL_MINOR_VERSION）
    set(Ascend_VERSION "")
    if(EXISTS "${Ascend_INCLUDE_DIR}/acl/acl_base.h")
        file(STRINGS "${Ascend_INCLUDE_DIR}/acl/acl_base.h" _ACL_VER_MAJOR
             REGEX "#define[ \t]+ACL_MAJOR_VERSION[ \t]+[0-9]+")
        file(STRINGS "${Ascend_INCLUDE_DIR}/acl/acl_base.h" _ACL_VER_MINOR
             REGEX "#define[ \t]+ACL_MINOR_VERSION[ \t]+[0-9]+")
        if(_ACL_VER_MAJOR MATCHES "ACL_MAJOR_VERSION[ \t]+([0-9]+)"
           AND _ACL_VER_MINOR MATCHES "ACL_MINOR_VERSION[ \t]+([0-9]+)")
            set(Ascend_VERSION "${CMAKE_MATCH_1}.${CMAKE_MATCH_2}")
        endif()
    endif()
    if(Ascend_VERSION)
        message(STATUS "Found Ascend CANN version: ${Ascend_VERSION}")
    endif()
    message(STATUS "Ascend ACL: ${Ascend_ACL_LIBRARY}")
endif()

mark_as_advanced(Ascend_INCLUDE_DIR Ascend_ACL_LIBRARY Ascend_ACL_DVPP_LIBRARY)
