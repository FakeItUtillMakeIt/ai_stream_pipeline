# cmake/FindEigen3.cmake
# ============================================================================
# FindEigen3 - 查找 Eigen3 库
# ============================================================================

# 优先使用项目自带的 Eigen
set(EIGEN3_ROOT_DIR ${CMAKE_SOURCE_DIR}/3rd_party CACHE PATH "Eigen3 root directory")

# 查找头文件
find_path(EIGEN3_INCLUDE_DIR
    NAMES Eigen/Core
    PATHS
        ${EIGEN3_ROOT_DIR}
        /usr/include/eigen3
        /usr/local/include/eigen3
        /opt/homebrew/include/eigen3
    DOC "Eigen3 include directory"
)

# 处理 REQUIRED 和 QUIET 参数
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Eigen3
    REQUIRED_VARS EIGEN3_INCLUDE_DIR
    VERSION_VAR EIGEN3_VERSION
)

# 创建导入目标
if(Eigen3_FOUND AND NOT TARGET Eigen3::Eigen)
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
    
    # 尝试获取版本号
    if(EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h")
        file(STRINGS "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" 
             _eigen_version_header
             REGEX "#define[ \t]+EIGEN_WORLD_VERSION[ \t]+")
        string(REGEX MATCH "[0-9]+" EIGEN_WORLD_VERSION "${_eigen_version_header}")
        
        file(STRINGS "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" 
             _eigen_version_header
             REGEX "#define[ \t]+EIGEN_MAJOR_VERSION[ \t]+")
        string(REGEX MATCH "[0-9]+" EIGEN_MAJOR_VERSION "${_eigen_version_header}")
        
        file(STRINGS "${EIGEN3_INCLUDE_DIR}/Eigen/src/Core/util/Macros.h" 
             _eigen_version_header
             REGEX "#define[ \t]+EIGEN_MINOR_VERSION[ \t]+")
        string(REGEX MATCH "[0-9]+" EIGEN_MINOR_VERSION "${_eigen_version_header}")
        
        set(EIGEN3_VERSION "${EIGEN_WORLD_VERSION}.${EIGEN_MAJOR_VERSION}.${EIGEN_MINOR_VERSION}")
    endif()
endif()

mark_as_advanced(EIGEN3_INCLUDE_DIR EIGEN3_ROOT_DIR)

message(STATUS "Eigen3: ${EIGEN3_INCLUDE_DIR} (version: ${EIGEN3_VERSION})")