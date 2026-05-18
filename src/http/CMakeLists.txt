# src/http/CMakeLists.txt

# 不需要 find_package(httplib REQUIRED)

add_executable(http_server
    api_server.cpp
    ${PROJECT_SOURCE_DIR}/3rd_party/log_mgr/log_mgr.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
)

target_link_libraries(http_server
    PRIVATE
        ai_stream_core
        ai_stream_nodes
        httplib::httplib          # FetchContent 提供的目标
        nlohmann_json::nlohmann_json
        spdlog::spdlog
)

target_include_directories(http_server
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

# 如果节点库使用了 whole-archive，这里要关闭
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(http_server PRIVATE -Wl,--no-whole-archive)
endif()