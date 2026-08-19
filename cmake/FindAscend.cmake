# cmake/FindAscend.cmake
# 查找华为 Ascend CANN SDK
#
# 定义以下变量：
#   Ascend_FOUND         - 是否找到 CANN SDK
#   Ascend_INCLUDE_DIRS  - 头文件目录
#   Ascend_LIBRARIES     - 库文件
#
# 以下变量可设置以指定路径：
#   ASCEND_HOME          - CANN SDK 安装根目录（如 /usr/local/Ascend/ascend-toolkit/latest）

find_path(Ascend_INCLUDE_DIR
    NAMES acl/acl.h
    PATHS
        ${ASCEND_HOME}/include
        /usr/local/Ascend/ascend-toolkit/latest/include
        /usr/local/Ascend/nnrt/latest/include
        $ENV{ASCEND_HOME}/include
)

find_library(Ascend_ACL_LIBRARY
    NAMES asccl
    PATHS
        ${ASCEND_HOME}/lib64
        /usr/local/Ascend/ascend-toolkit/latest/lib64
        /usr/local/Ascend/nnrt/latest/lib64
        $ENV{ASCEND_HOME}/lib64
)

find_library(Ascend_ACL_DVPP_LIBRARY
    NAMES acl_dvpp
    PATHS
        ${ASCEND_HOME}/lib64
        /usr/local/Ascend/ascend-toolkit/latest/lib64
        /usr/local/Ascend/nnrt/latest/lib64
        $ENV{ASCEND_HOME}/lib64
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Ascend
    REQUIRED_VARS Ascend_ACL_LIBRARY Ascend_INCLUDE_DIR
    FAIL_MESSAGE "Ascend CANN SDK not found. Set ASCEND_HOME or install CANN SDK."
)

if(Ascend_FOUND)
    set(Ascend_INCLUDE_DIRS ${Ascend_INCLUDE_DIR})
    set(Ascend_LIBRARIES ${Ascend_ACL_LIBRARY})
    if(Ascend_ACL_DVPP_LIBRARY)
        list(APPEND Ascend_LIBRARIES ${Ascend_ACL_DVPP_LIBRARY})
    endif()
endif()

mark_as_advanced(Ascend_INCLUDE_DIR Ascend_ACL_LIBRARY Ascend_ACL_DVPP_LIBRARY)
