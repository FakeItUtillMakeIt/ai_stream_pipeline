# cmake/FindTensorRT.cmake
# 查找 TensorRT 安装路径
find_path(TensorRT_INCLUDE_DIR NvInfer.h
    PATHS /usr/include/x86_64-linux-gnu
          /usr/local/tensorrt/include
          $ENV{TENSORRT_DIR}/include
)
find_library(TensorRT_LIBRARY NAMES nvinfer
    PATHS /usr/lib/x86_64-linux-gnu
          /usr/local/tensorrt/lib
          $ENV{TENSORRT_DIR}/lib
)
if(TensorRT_INCLUDE_DIR AND TensorRT_LIBRARY)
    set(TensorRT_FOUND TRUE)
    set(TensorRT_INCLUDE_DIRS ${TensorRT_INCLUDE_DIR})
    set(TensorRT_LIBRARIES ${TensorRT_LIBRARY})

    # 从 NvInferVersion.h 提取版本（NV_TENSORRT_MAJOR / NV_TENSORRT_MINOR / NV_TENSORRT_PATCH）
    set(TensorRT_VERSION "")
    set(_TRT_VER_HEADER "${TensorRT_INCLUDE_DIR}/NvInferVersion.h")
    if(NOT EXISTS "${_TRT_VER_HEADER}")
        set(_TRT_VER_HEADER "${TensorRT_INCLUDE_DIR}/NvInfer.h")
    endif()
    if(EXISTS "${_TRT_VER_HEADER}")
        foreach(_comp MAJOR MINOR PATCH BUILD)
            # NV_TENSORRT_x 的值可能是数字或间接引用另一宏（如企业版 TRT_MAJOR_ENTERPRISE）
            file(STRINGS "${_TRT_VER_HEADER}" _TRT_V_${_comp}
                 REGEX "#define[ \t]+NV_TENSORRT_${_comp}[ \t]+[A-Za-z0-9_]+")
            set(_val "")
            if(_TRT_V_${_comp} MATCHES "NV_TENSORRT_${_comp}[ \t]+([A-Za-z0-9_]+)")
                set(_val "${CMAKE_MATCH_1}")
                if(NOT _val MATCHES "^[0-9]+$")
                    # 间接引用：再查一次该宏的数值
                    file(STRINGS "${_TRT_VER_HEADER}" _TRT_REF
                         REGEX "#define[ \t]+${_val}[ \t]+[0-9]+")
                    if(_TRT_REF MATCHES "#define[ \t]+${_val}[ \t]+([0-9]+)")
                        set(_val "${CMAKE_MATCH_1}")
                    else()
                        set(_val "")
                    endif()
                endif()
            endif()
            if(_val)
                if(TensorRT_VERSION)
                    string(APPEND TensorRT_VERSION ".${_val}")
                else()
                    set(TensorRT_VERSION "${_val}")
                endif()
            endif()
        endforeach()
    endif()
    if(TensorRT_VERSION)
        message(STATUS "Found TensorRT version: ${TensorRT_VERSION}")
    endif()
endif()
mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY)