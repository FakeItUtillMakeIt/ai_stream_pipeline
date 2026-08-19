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

find_path(RKNN_INCLUDE_DIR
    NAMES rknn_api.h
    PATHS
        ${RKNN_ROOT}/include
        /usr/include
        /usr/local/include
        /opt/rknn/include
        $ENV{RKNN_ROOT}/include
)

find_library(RKNN_LIBRARY
    NAMES rknn_api
    PATHS
        ${RKNN_ROOT}/lib
        /usr/lib
        /usr/local/lib
        /opt/rknn/lib
        $ENV{RKNN_ROOT}/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RKNN
    REQUIRED_VARS RKNN_LIBRARY RKNN_INCLUDE_DIR
    FAIL_MESSAGE "RKNN SDK not found. Set RKNN_ROOT or install RKNN SDK."
)

if(RKNN_FOUND)
    set(RKNN_INCLUDE_DIRS ${RKNN_INCLUDE_DIR})
    set(RKNN_LIBRARIES ${RKNN_LIBRARY})
endif()

mark_as_advanced(RKNN_INCLUDE_DIR RKNN_LIBRARY)
