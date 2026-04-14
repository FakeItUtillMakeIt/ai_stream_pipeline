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
endif()
mark_as_advanced(TensorRT_INCLUDE_DIR TensorRT_LIBRARY)