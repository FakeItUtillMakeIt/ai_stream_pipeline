# cmake/aarch64-rk3588.toolchain.cmake
# RK3588 交叉编译工具链
#
# 依赖两个路径（按实际情况传入）：
#   RK3588_SYSROOT        目标平台 sysroot（含 OpenCV/FFmpeg 等目标架构依赖，
#                         通常取自 Rockchip SDK 的 buildroot/debian sysroot）
#   RK_TOOLCHAIN_PREFIX   交叉工具链前缀（可选）：
#                         - 系统交叉编译器: /usr/bin/aarch64-linux-gnu-
#                         - Rockchip SDK 自带: <sdk>/prebuilts/gcc/.../bin/aarch64-buildroot-linux-gnu-
#
# 用法示例：
#   cmake -B build-rk3588 \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-rk3588.toolchain.cmake \
#     -DRK3588_SYSROOT=/path/to/sdk/sysroot \
#     -DRK_TOOLCHAIN_PREFIX=/usr/bin/aarch64-linux-gnu- \
#     -DWITH_RKNN=ON -DWITH_CUDA=OFF -DWITH_TENSORRT=OFF \
#     -DCMAKE_BUILD_TYPE=Release
#
# RKNN/RGA/MPP 的 aarch64 库已内置在 3rd_party/rk_platform/，
# 无需在 sysroot 中另装；如需指定其它 SDK 路径可加 -DRKNN_ROOT=<sdk>/rknn。

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

if(NOT RK_TOOLCHAIN_PREFIX)
    set(RK_TOOLCHAIN_PREFIX "/usr/bin/aarch64-linux-gnu-")
endif()

set(CMAKE_C_COMPILER   "${RK_TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${RK_TOOLCHAIN_PREFIX}g++")
set(CMAKE_AR           "${RK_TOOLCHAIN_PREFIX}ar")
set(CMAKE_RANLIB       "${RK_TOOLCHAIN_PREFIX}ranlib")
set(CMAKE_STRIP        "${RK_TOOLCHAIN_PREFIX}strip")

# 目标 sysroot：OpenCV / FFmpeg / Eigen 等从目标架构环境查找
if(RK3588_SYSROOT)
    set(CMAKE_SYSROOT "${RK3588_SYSROOT}")
    set(CMAKE_FIND_ROOT_PATH "${RK3588_SYSROOT}")
else()
    message(WARNING
        "RK3588_SYSROOT 未设置：将只使用工具链默认搜索路径。"
        "若主机上没有目标架构的 OpenCV/FFmpeg，配置会失败。")
endif()

# 工程内置的 aarch64 三方库（rknn/rga/mpp）同样参与查找
list(APPEND CMAKE_FIND_ROOT_PATH "${CMAKE_CURRENT_LIST_DIR}/../3rd_party")

# 头文件/库只在 sysroot 与 3rd_party 中查找；工具（如编译期生成的可执行）
# 用主机版本
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE BOTH)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE BOTH)

# aarch64 必要编译选项
set(CMAKE_C_FLAGS_INIT   "-fPIC")
set(CMAKE_CXX_FLAGS_INIT "-fPIC")

# 防止 CMake 探测把主机特性带入目标
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
