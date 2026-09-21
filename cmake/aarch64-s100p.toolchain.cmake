# cmake/aarch64-s100p.toolchain.cmake
# D-Robotics S100P (RDK) 交叉编译工具链
#
# 依赖：
#   S100P_SYSROOT        目标平台 sysroot（从 S100P 板子导出）
#
# 用法：
#   cmake -B build-s100p \
#     -DCMAKE_TOOLCHAIN_FILE=cmake/aarch64-s100p.toolchain.cmake \
#     -DS100P_SYSROOT=/opt/ci-assets/s100p-sysroot \
#     -DWITH_HORIZON=ON -DWITH_CUDA=OFF -DWITH_TENSORRT=OFF \
#     -DCMAKE_BUILD_TYPE=Release
#   cmake --build build-s100p -j$(nproc)

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR aarch64)

# 交叉编译器
set(CMAKE_C_COMPILER   aarch64-linux-gnu-gcc)
set(CMAKE_CXX_COMPILER aarch64-linux-gnu-g++)
set(CMAKE_AR           aarch64-linux-gnu-ar)
set(CMAKE_RANLIB       aarch64-linux-gnu-ranlib)
set(CMAKE_STRIP        aarch64-linux-gnu-strip)

# 目标 sysroot
if(S100P_SYSROOT)
    set(CMAKE_SYSROOT "${S100P_SYSROOT}")
    set(CMAKE_FIND_ROOT_PATH "${S100P_SYSROOT}")
    # Debian multiarch：OpenCV/FFmpeg 的 cmake config 在 lib/aarch64-linux-gnu/cmake 下
    list(APPEND CMAKE_PREFIX_PATH
        "${S100P_SYSROOT}/usr/lib/aarch64-linux-gnu/cmake"
        "${S100P_SYSROOT}/usr/lib/aarch64-linux-gnu"
        "${S100P_SYSROOT}/usr/local/lib/cmake"
    )
    # 头文件搜索路径
    list(APPEND CMAKE_INCLUDE_PATH "${S100P_SYSROOT}/usr/include/aarch64-linux-gnu")
    # 库文件搜索路径
    list(APPEND CMAKE_LIBRARY_PATH "${S100P_SYSROOT}/usr/lib/aarch64-linux-gnu")
    # 链接器 rpath-link
    set(CMAKE_EXE_LINKER_FLAGS_INIT
        "-Wl,-rpath-link,${S100P_SYSROOT}/usr/lib/aarch64-linux-gnu -Wl,-rpath-link,${S100P_SYSROOT}/lib/aarch64-linux-gnu")
else()
    message(WARNING
        "S100P_SYSROOT 未设置：将只使用工具链默认搜索路径。"
        "若主机上没有目标架构的 OpenCV/FFmpeg，配置会失败。")
endif()

# 工程内置的三方库路径
list(APPEND CMAKE_FIND_ROOT_PATH "${CMAKE_CURRENT_LIST_DIR}/../3rd_party")

# 头文件/库只在 sysroot 与 3rd_party 中查找；工具用主机版本
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# aarch64 编译选项
set(CMAKE_C_FLAGS_INIT   "-fPIC -march=armv8.2-a")
set(CMAKE_CXX_FLAGS_INIT "-fPIC -march=armv8.2-a")

# 防止 CMake 探测把主机特性带入目标
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
