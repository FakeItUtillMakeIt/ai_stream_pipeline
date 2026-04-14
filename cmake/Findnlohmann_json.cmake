# cmake/Findnlohmann_json.cmake

# 直接找到包含 nlohmann 文件夹的父目录
get_filename_component(nlohmann_json_INCLUDE_DIR 
    "${PROJECT_SOURCE_DIR}/3rd_party" 
    ABSOLUTE
)

# 验证文件是否存在
if(EXISTS "${nlohmann_json_INCLUDE_DIR}/nlohmann/json.hpp")
    set(nlohmann_json_FOUND TRUE)
    message(STATUS "Found nlohmann_json: ${nlohmann_json_INCLUDE_DIR}")
else()
    set(nlohmann_json_FOUND FALSE)
    message(FATAL_ERROR "Could not find nlohmann/json.hpp in ${nlohmann_json_INCLUDE_DIR}")
endif()

# 创建接口库
if(nlohmann_json_FOUND AND NOT TARGET nlohmann_json::nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED)
    set_target_properties(nlohmann_json::nlohmann_json PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${nlohmann_json_INCLUDE_DIR}"
    )
endif()