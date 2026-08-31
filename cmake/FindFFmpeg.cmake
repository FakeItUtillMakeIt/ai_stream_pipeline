# cmake/FindFFmpeg.cmake

# 查找 FFmpeg 库的各个组件
find_path(AVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES ffmpeg
    HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT}
)

find_path(AVFORMAT_INCLUDE_DIR
    NAMES libavformat/avformat.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES ffmpeg
    HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT}
)

find_path(AVUTIL_INCLUDE_DIR
    NAMES libavutil/avutil.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES ffmpeg
    HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT}
)

find_path(SWSCALE_INCLUDE_DIR
    NAMES libswscale/swscale.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES ffmpeg
    HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT}
)

find_path(SWRESAMPLE_INCLUDE_DIR
    NAMES libswresample/swresample.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES ffmpeg
)

# 查找库文件
find_library(AVCODEC_LIBRARY NAMES avcodec HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT})
find_library(AVFORMAT_LIBRARY NAMES avformat HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT})
find_library(AVUTIL_LIBRARY NAMES avutil HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT})
find_library(SWSCALE_LIBRARY NAMES swscale HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT})
find_library(SWRESAMPLE_LIBRARY NAMES swresample HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT})

# 处理 REQUIRED 参数
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(FFmpeg
    REQUIRED_VARS
        AVCODEC_INCLUDE_DIR AVCODEC_LIBRARY
        AVFORMAT_INCLUDE_DIR AVFORMAT_LIBRARY
        AVUTIL_INCLUDE_DIR AVUTIL_LIBRARY
        SWSCALE_INCLUDE_DIR SWSCALE_LIBRARY
)

set(FFMPEG_LIBRARIES 
    ${AVCODEC_LIBRARY}
    ${AVFORMAT_LIBRARY}
    ${AVUTIL_LIBRARY}
    ${SWSCALE_LIBRARY}
)

# 设置头文件路径
set(FFMPEG_INCLUDE_DIRS
    ${AVCODEC_INCLUDE_DIR}
    ${AVFORMAT_INCLUDE_DIR}
    ${AVUTIL_INCLUDE_DIR}
    ${SWSCALE_INCLUDE_DIR}
)

# 创建导入目标（如果找到）
if(FFmpeg_FOUND AND NOT TARGET FFmpeg::FFmpeg)
    add_library(FFmpeg::FFmpeg INTERFACE IMPORTED)
    target_include_directories(FFmpeg::FFmpeg INTERFACE
        ${AVCODEC_INCLUDE_DIR}
        ${AVFORMAT_INCLUDE_DIR}
        ${AVUTIL_INCLUDE_DIR}
        ${SWSCALE_INCLUDE_DIR}
    )
    target_link_libraries(FFmpeg::FFmpeg INTERFACE
        ${AVCODEC_LIBRARY}
        ${AVFORMAT_LIBRARY}
        ${AVUTIL_LIBRARY}
        ${SWSCALE_LIBRARY}
        ${SWRESAMPLE_LIBRARY}
    )
endif()

mark_as_advanced(
    AVCODEC_INCLUDE_DIR AVFORMAT_INCLUDE_DIR AVUTIL_INCLUDE_DIR
    SWSCALE_INCLUDE_DIR SWRESAMPLE_INCLUDE_DIR
    AVCODEC_LIBRARY AVFORMAT_LIBRARY AVUTIL_LIBRARY
    SWSCALE_LIBRARY SWRESAMPLE_LIBRARY
)
