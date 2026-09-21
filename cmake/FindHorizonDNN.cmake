# cmake/FindHorizonDNN.cmake
# 查找地平线 DNN SDK（RDK S100P / Horizon BPU）
#
# 定义以下变量：
#   HORIZON_DNN_FOUND       - 是否找到 Horizon DNN SDK
#   HORIZON_DNN_INCLUDE_DIRS - 头文件目录
#   HORIZON_DNN_LIBRARIES    - 库文件
#
# 以下变量可设置以指定路径：
#   HORIZON_DNN_ROOT        - SDK 安装根目录

find_path(HORIZON_DNN_INCLUDE_DIR
    NAMES hb_dnn.h
    PATHS
        ${HORIZON_DNN_ROOT}/include
        ${CMAKE_SYSROOT}/usr/include/hobot/dnn
        ${CMAKE_SYSROOT}/usr/include/hobot
        ${CMAKE_SYSROOT}/usr/hobot/include
        /usr/include/hobot/dnn
        /usr/include/hobot
        /usr/hobot/include
        /usr/local/include
        /opt/hobot/include
        $ENV{HORIZON_DNN_ROOT}/include
)

find_library(HORIZON_DNN_LIBRARY
    NAMES dnn hb_dnn
    PATHS
        ${HORIZON_DNN_ROOT}/lib
        ${CMAKE_SYSROOT}/usr/hobot/lib
        /usr/hobot/lib
        /usr/lib
        /usr/local/lib
        /opt/hobot/lib
        $ENV{HORIZON_DNN_ROOT}/lib
)

find_library(HORIZON_UCP_LIBRARY
    NAMES hbucp ucp
    PATHS
        ${HORIZON_DNN_ROOT}/lib
        ${CMAKE_SYSROOT}/usr/hobot/lib
        /usr/hobot/lib
        /usr/lib
        /usr/local/lib
        /opt/hobot/lib
        $ENV{HORIZON_DNN_ROOT}/lib
)

# BPU 库（可选，用于直接 BPU 控制）
find_library(HORIZON_BPU_LIBRARY
    NAMES bpu hb_bpu
    PATHS
        ${HORIZON_DNN_ROOT}/lib
        ${CMAKE_SYSROOT}/usr/hobot/lib
        /usr/hobot/lib
        /usr/lib
        /usr/local/lib
        /opt/hobot/lib
        $ENV{HORIZON_DNN_ROOT}/lib
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(HorizonDNN
    REQUIRED_VARS HORIZON_DNN_LIBRARY HORIZON_DNN_INCLUDE_DIR
    FAIL_MESSAGE "Horizon DNN SDK not found. Set HORIZON_DNN_ROOT or install hobot-dnn package."
)

# find_package_handle_standard_args sets <PackageName>_FOUND (HorizonDNN_FOUND)
# Also set HORIZON_DNN_FOUND for compatibility with CMakeLists.txt
if(HorizonDNN_FOUND)
    set(HORIZON_DNN_FOUND TRUE)
endif()

if(HORIZON_DNN_FOUND)
    set(HORIZON_DNN_INCLUDE_DIRS ${HORIZON_DNN_INCLUDE_DIR})
    # hb_ucp.h / hb_ucp_sys.h are in the parent directory of hb_dnn.h
    get_filename_component(_HORIZON_DNN_PARENT_DIR "${HORIZON_DNN_INCLUDE_DIR}" DIRECTORY)
    list(APPEND HORIZON_DNN_INCLUDE_DIRS ${_HORIZON_DNN_PARENT_DIR})
    set(HORIZON_DNN_LIBRARIES ${HORIZON_DNN_LIBRARY})

    if(HORIZON_UCP_LIBRARY)
        list(APPEND HORIZON_DNN_LIBRARIES ${HORIZON_UCP_LIBRARY})
    endif()

    if(HORIZON_BPU_LIBRARY)
        list(APPEND HORIZON_DNN_LIBRARIES ${HORIZON_BPU_LIBRARY})
    endif()

    # 从 hb_ucp.h 提取 SDK 版本
    set(HORIZON_DNN_VERSION "")
    if(EXISTS "${HORIZON_DNN_INCLUDE_DIR}/hb_ucp.h")
        file(STRINGS "${HORIZON_DNN_INCLUDE_DIR}/hb_ucp.h" _HB_VER_MAJOR
             REGEX "#define[ \t]+HB_UCP_VERSION_MAJOR[ \t]+\\(([0-9]+)U\\)")
        file(STRINGS "${HORIZON_DNN_INCLUDE_DIR}/hb_ucp.h" _HB_VER_MINOR
             REGEX "#define[ \t]+HB_UCP_VERSION_MINOR[ \t]+\\(([0-9]+)U\\)")
        file(STRINGS "${HORIZON_DNN_INCLUDE_DIR}/hb_ucp.h" _HB_VER_PATCH
             REGEX "#define[ \t]+HB_UCP_VERSION_PATCH[ \t]+\\(([0-9]+)U\\)")
        if(_HB_VER_MAJOR MATCHES "HB_UCP_VERSION_MAJOR[ \t]+\\(([0-9]+)U\\)")
            set(_HB_MAJ "${CMAKE_MATCH_1}")
        endif()
        if(_HB_VER_MINOR MATCHES "HB_UCP_VERSION_MINOR[ \t]+\\(([0-9]+)U\\)")
            set(_HB_MIN "${CMAKE_MATCH_1}")
        endif()
        if(_HB_VER_PATCH MATCHES "HB_UCP_VERSION_PATCH[ \t]+\\(([0-9]+)U\\)")
            set(_HB_PAT "${CMAKE_MATCH_1}")
        endif()
        if(_HB_MAJ AND _HB_MIN AND _HB_PAT)
            set(HORIZON_DNN_VERSION "${_HB_MAJ}.${_HB_MIN}.${_HB_PAT}")
        endif()
    endif()

    if(HORIZON_DNN_VERSION)
        message(STATUS "Found Horizon DNN SDK version: ${HORIZON_DNN_VERSION}")
    endif()

    # 平台一致性提示
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        message(STATUS "Target platform: aarch64 (Horizon BPU)")
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
        message(WARNING "Target platform: x86_64; Horizon BPU backend will be compiled but dlopen will fail at runtime")
    endif()
endif()

mark_as_advanced(HORIZON_DNN_INCLUDE_DIR HORIZON_DNN_LIBRARY HORIZON_UCP_LIBRARY HORIZON_BPU_LIBRARY)
