# src/http/CMakeLists.txt

# 不需要 find_package(httplib REQUIRED)

add_executable(http_server
    api_server.cpp
    request_log.cpp
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

# libcurl 仅在 WITH_FTP/WITH_VLM 打开且找到可用 curl 时才链接；
# api_server 本身不直接调用 curl，缺失时不应阻断 HTTP 服务构建。
if(TARGET CURL::libcurl)
    target_link_libraries(http_server PRIVATE CURL::libcurl)
endif()

target_include_directories(http_server
    PRIVATE
        ${PROJECT_SOURCE_DIR}
)

# --no-as-needed：ai_stream_nodes 靠静态 REGISTER_NODE 注册节点，无直接符号引用，
# 否则会被链接器裁掉导致节点工厂为空、无法创建任何节点
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_link_options(http_server PRIVATE -Wl,--no-as-needed -Wl,--no-whole-archive)
endif()