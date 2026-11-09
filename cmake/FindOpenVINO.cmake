# cmake/FindOpenVINO.cmake
# 查找 OpenVINO C API（openvino/c/openvino.h + libopenvino_c）
#
# 支持三种安装形态：
#   1. 官方 OpenVINO Runtime（/opt/intel/openvino/runtime）
#   2. 系统包管理器安装
#   3. pip wheel（python3 -c "import openvino"）
#      wheel 内含 C++/C 头文件与 libs/，C API 走 C ABI，不受 C++ ABI 影响。
#
# 输入变量：
#   OPENVINO_ROOT            可选，显式指定 OpenVINO 安装根目录
#   OPENVINO_INCLUDE_DIR     可选，直接指定头文件目录
# 输出变量：
#   OpenVINO_FOUND / OPENVINO_FOUND
#   OPENVINO_INCLUDE_DIRS
#   OPENVINO_LIBRARIES
#   OPENVINO_LIBRARY_DIR

set(_ov_hints "")
if(OPENVINO_ROOT)
    list(APPEND _ov_hints "${OPENVINO_ROOT}")
endif()
if(DEFINED ENV{OPENVINO_ROOT})
    list(APPEND _ov_hints "$ENV{OPENVINO_ROOT}")
endif()
if(DEFINED ENV{OPENVINO_DIR})
    list(APPEND _ov_hints "$ENV{OPENVINO_DIR}")
endif()

list(APPEND _ov_hints
    /opt/intel/openvino
    /opt/intel/openvino/runtime
    /opt/openvino
    /usr/local
    /usr
)

# pip wheel：openvino/__init__.py 同级目录下有 include/ 与 libs/
if(NOT OPENVINO_ROOT)
    find_program(_OV_PYTHON3 NAMES python3 python)
    if(_OV_PYTHON3)
        execute_process(
            COMMAND ${_OV_PYTHON3} -c "import openvino, os; print(os.path.dirname(openvino.__file__))"
            OUTPUT_VARIABLE _ov_pip_dir
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE _ov_pip_ret
        )
        if(_ov_pip_ret EQUAL 0 AND _ov_pip_dir)
            list(APPEND _ov_hints "${_ov_pip_dir}")
        endif()
    endif()
endif()

find_path(OPENVINO_INCLUDE_DIR
    NAMES openvino/c/openvino.h
    HINTS ${_ov_hints}
    PATH_SUFFIXES include runtime/include
)

# find_library 对只有版本号后缀（libopenvino_c.so.2460）的 wheel 不可靠，
# 因此在候选目录中直接 glob。
set(_ov_lib_dirs "")
foreach(_hint ${_ov_hints})
    list(APPEND _ov_lib_dirs
        "${_hint}/libs"
        "${_hint}/runtime/lib/intel64"
        "${_hint}/lib"
        "${_hint}/lib/intel64"
        "${_hint}/lib/x86_64-linux-gnu"
    )
endforeach()

set(_ov_c_lib "")
foreach(_dir ${_ov_lib_dirs})
    if(EXISTS "${_dir}")
        file(GLOB _ov_candidates "${_dir}/libopenvino_c.so" "${_dir}/libopenvino_c.so.*")
        if(_ov_candidates)
            list(SORT _ov_candidates)
            list(GET _ov_candidates 0 _ov_c_lib)
            break()
        endif()
    endif()
endforeach()

if(NOT _ov_c_lib)
    find_library(_ov_c_lib NAMES openvino_c HINTS ${_ov_hints}
        PATH_SUFFIXES libs runtime/lib/intel64 lib lib/intel64)
endif()

if(OPENVINO_INCLUDE_DIR AND _ov_c_lib)
    set(OPENVINO_FOUND TRUE)
    set(OpenVINO_FOUND TRUE)
    set(OPENVINO_INCLUDE_DIRS "${OPENVINO_INCLUDE_DIR}")
    set(OPENVINO_LIBRARIES "${_ov_c_lib}")
    get_filename_component(OPENVINO_LIBRARY_DIR "${_ov_c_lib}" DIRECTORY)
    message(STATUS "OpenVINO: ${_ov_c_lib}")
else()
    set(OPENVINO_FOUND FALSE)
    set(OpenVINO_FOUND FALSE)
    if(OpenVINO_FIND_REQUIRED)
        message(FATAL_ERROR "OpenVINO not found. Set OPENVINO_ROOT or install the OpenVINO runtime.")
    endif()
endif()

mark_as_advanced(OPENVINO_INCLUDE_DIR OPENVINO_LIBRARIES OPENVINO_LIBRARY_DIR)
